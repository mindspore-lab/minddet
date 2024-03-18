"""PointPillarsNet"""
import numpy as np
from mindspore import context, nn
from mindspore import numpy as mnp
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.ops.primitive import constexpr
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from src.core.losses import (
    SigmoidFocalClassificationLoss,
    WeightedSmoothL1LocalizationLoss,
    WeightedSoftmaxClassificationLoss,
)


def prepare_loss_weights(
    labels, pos_cls_weight=1.0, neg_cls_weight=1.0, dtype=mstype.float32
):
    """get cls_weights and reg_weights from labels."""
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.astype(dtype)
    reg_weights = positives.astype(dtype)
    pos_normalizer = (
        positives.astype(mstype.float16).sum(1, keepdims=True).astype(dtype)
    )
    reg_weights /= ops.clip_by_value(
        pos_normalizer,
        clip_value_min=_create_on_value(),
        clip_value_max=pos_normalizer.max(),
    )
    cls_weights /= ops.clip_by_value(
        pos_normalizer,
        clip_value_min=_create_on_value(),
        clip_value_max=pos_normalizer.max(),
    )
    return cls_weights, reg_weights, cared


@constexpr
def _create_on_value():
    """create on value"""
    return Tensor(1.0, mstype.float32)


@constexpr
def _create_off_value():
    """create off value"""
    return Tensor(0.0, mstype.float32)


@constexpr
def _log16():
    """log(16)"""
    return ops.Log()(Tensor(16.0, mstype.float32))


def create_loss(
    loc_loss_ftor,
    cls_loss_ftor,
    box_preds,
    cls_preds,
    cls_targets,
    cls_weights,
    reg_targets,
    reg_weights,
    num_class,
    encode_background_as_zeros=True,
    encode_rad_error_by_sin=True,
    box_code_size=7,
):
    """create loss"""
    batch_size = box_preds.shape[0]
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = ops.OneHot()(
        cls_targets, num_class + 1, _create_on_value(), _create_off_value()
    )
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights
    )  # [N, M]
    return loc_losses, cls_losses


def add_sin_difference(boxes1, boxes2):
    """add sin difference"""
    rad_pred_encoding = ops.Sin()(boxes1[..., -1:]) * ops.Cos()(boxes2[..., -1:])
    rad_tg_encoding = ops.Cos()(boxes1[..., -1:]) * ops.Sin()(boxes2[..., -1:])
    boxes1 = ops.Concat(axis=-1)([boxes1[..., :-1], rad_pred_encoding])
    boxes2 = ops.Concat(axis=-1)([boxes2[..., :-1], rad_tg_encoding])
    return boxes1, boxes2


def _get_pos_neg_loss(cls_loss, labels):
    """get pos neg loss"""
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).astype(cls_loss.dtype) * cls_loss.view(
            batch_size, -1
        )
        cls_neg_loss = (labels == 0).astype(cls_loss.dtype) * cls_loss.view(
            batch_size, -1
        )
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return ops.Tensor.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - ops.floor(val / period + offset) * period
    return ans.asnumpy() if is_numpy else ans


def get_direction_target(anchors, reg_targets, one_hot=True, use_self_train=True):
    """get direction target"""
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    num_bins = 2
    if not use_self_train:
        dir_offset = 0.78539
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = ops.floor(offset_rot / (2 * np.pi / num_bins))
        dir_cls_targets = ops.clip_by_value(
            dir_cls_targets, clip_value_min=0, clip_value_max=(num_bins - 1)
        )
    else:
        dir_cls_targets = (ops.greater(rot_gt, 0)).astype(mstype.int64)
    if one_hot:
        dir_cls_targets = ops.OneHot()(
            dir_cls_targets.astype(mstype.int32),
            num_bins,
            _create_on_value(),
            _create_off_value(),
        )
    return dir_cls_targets


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor"""

    actual_num = ops.ExpandDims()(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = mnp.arange(0, max_num, dtype=mstype.int32).view(*max_num_shape)
    paddings_indicator = actual_num > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Cell):
    """PFN layer"""

    def __init__(self, in_channels, out_channels, use_norm, last_layer):
        super().__init__()

        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.units = out_channels
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.99)
        else:
            self.norm = ops.Identity()
        self.linear = nn.Dense(
            in_channels, self.units, bias_init="normal", has_bias=not use_norm
        )
        self.linear.to_float(mstype.float16)
        self.transpose = ops.Transpose()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=2)
        self.expand_dims = ops.ExpandDims()
        self.argmax_w_value = ops.ReduceMax(keep_dims=True)
        self.stack = ops.Stack()

    def construct(self, inputs):
        """forward graph"""
        # (batch_size,voxel_size,point_num,10)
        x = self.linear(inputs)
        x = self.norm(x.transpose((0, 3, 1, 2))).transpose(
            (0, 2, 3, 1)
        )  # [bs, V, P, 4]
        # (batch_size, voxel_size, point_num, 64)
        x = ops.ReLU()(x)
        # [batch_size, voxel_size, 64)]
        x_max = self.argmax_w_value(x, 2)
        if self.last_vfe:
            return x_max
        x_repeat = self.tile(x_max, (1, 1, inputs.shape[1], 1))  # [bs, V, 64]
        x_concatenated = self.concat([x, x_repeat])
        return x_concatenated


class PillarFeatureNet(nn.Cell):
    """Pillar feature net"""

    def __init__(
        self,
        num_input_features=4,
        use_norm=True,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
    ):
        super().__init__()
        self.use_absolute_xyz = True
        num_input_features += 6 if self.use_absolute_xyz else 3

        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)  # (10,64)
        pfn_layers = []

        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True

            pfn_layers.append(
                PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer)
            )
        self.pfn_layers = nn.SequentialCell(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.z_offset = self.vz / 2 + pc_range[2]
        self.expand_dims = ops.ExpandDims()

    def construct(self, features, num_points, coors):
        """forward graph"""
        # features [batch_size, voxel_num, point_num, coord_dim]
        # features(4,40000,32,4)   num_points(4,40000)   coors(4,40000,4)
        features = features.astype(mstype.float32)
        bs, v, _, _ = features.shape
        # (bs,voxel,1,3)
        points_mean = features[:, :, :, :3].sum(axis=2, keepdims=True) / ops.Maximum()(
            num_points, 1
        ).view(bs, v, 1, 1)
        f_cluster = features[:, :, :, :3] - points_mean
        f_center0 = features[:, :, :, 0] - (
            self.expand_dims(coors[:, :, 2].astype(mstype.float32), 2) * self.vx
            + self.x_offset
        )
        f_center1 = features[:, :, :, 1] - (
            self.expand_dims(coors[:, :, 1].astype(mstype.float32), 2) * self.vy
            + self.y_offset
        )
        f_center2 = features[:, :, :, 2] - (
            self.expand_dims(coors[:, :, 0].astype(mstype.float32), 2) * self.vz
            + self.z_offset
        )

        f_center0 = ops.ExpandDims()(f_center0, 3)
        f_center1 = ops.ExpandDims()(f_center1, 3)
        f_center2 = ops.ExpandDims()(f_center2, 3)
        f_center = ops.Concat(axis=-1)((f_center0, f_center1, f_center2)).astype(
            features.dtype
        )

        # Combine feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = mnp.norm(features[:, :, :, :3], 2, 3, keepdims=True)
            features_ls.append(points_dist)
        features = ops.Concat(axis=-1)(features_ls)
        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zero.
        voxel_count = features.shape[2]
        mask = get_paddings_indicator(num_points, voxel_count, axis=1)
        mask = self.expand_dims(mask, -1).astype(features.dtype)
        features *= mask
        # Forward pass through PFNLayers
        features = self.pfn_layers(features)
        return features.squeeze(axis=2)


# Generate Pseudo-Image
class PointPillarsScatter(nn.Cell):
    """PointPillars scatter"""

    def __init__(self, output_shape, num_input_features):
        super().__init__()
        self.output_shape = output_shape
        self.nz = output_shape[1]
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        # [432,496,1]
        assert self.nz == 1
        self.n_channels = num_input_features
        self.scatter_nd = ops.ScatterNd()
        self.concat = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.zeros = ops.Zeros()

    def construct(self, voxel_features, coords):
        """forward graph"""
        # Batch_canvas will be the final output.  (batch_size,voxel_size,point_num,64)
        batch_size = voxel_features.shape[0]  # [bs, v, 64]
        # [bs, v, 64?]", voxel_features.shape)
        coords = coords.astype(mstype.float32)  # [bs,v,4]  ->[z,y,x,1or0]

        #  z coordinate is not used, z -> batch
        for i in range(batch_size):
            coords[i, :, 0] = i
        shape = (
            batch_size,
            self.ny,
            self.nx,
            2,
            self.n_channels,
        )  # [bs, 496, 432, 2, 64]
        batch_canvas = self.scatter_nd(
            coords.astype(mstype.int32), voxel_features, shape
        )  # [bs, v, p, 2, 64]
        # [bs, 496, 432, 2, 64]
        batch_canvas = batch_canvas[:, :, :, 1]
        # [4 496 432 64])
        batch_canvas = self.transpose(batch_canvas, (0, 3, 1, 2))
        # batch_canvas.shape = [4 64 496 432])
        return batch_canvas


class RPN(nn.Cell):
    """RPN"""

    def __init__(
        self,
        box_coder,
        use_self_train=True,
        use_norm=True,
        num_class=2,
        layer_nums=(3, 5, 5),
        layer_strides=(2, 2, 2),
        num_filters=(128, 128, 256),
        upsample_strides=(1, 2, 4),
        num_upsample_filters=(256, 256, 256),
        num_input_filters=128,
        num_anchor_per_loc=2,
        encode_background_as_zeros=True,
        use_direction_classifier=True,
        use_bev=False,
        box_code_size=7,
    ):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self.use_direction_classifier = use_direction_classifier
        self.use_bev = use_bev
        self._use_norm = use_norm
        self.box_coder = box_coder
        self.encode_background_as_zeros = encode_background_as_zeros
        self.num_class = num_class
        self.use_self_train = use_self_train

        if len(layer_nums) != 3:
            raise ValueError(f"Layer nums must be 3, got {layer_nums}")
        if len(layer_nums) != len(layer_strides):
            raise ValueError(
                f"Layer nums and layer strides must have same length, "
                f"got {len(layer_nums)}, {len(layer_strides)} rescpectively"
            )
        if len(layer_nums) != len(num_filters):
            raise ValueError(
                f"Layer nums and num filters must have same length, "
                f"got {len(layer_nums)}, {len(num_filters)} respectively"
            )
        if len(layer_nums) != len(upsample_strides):
            raise ValueError(
                f"Layer nums and upsample strides must have same length, "
                f"got {len(layer_nums)}, {len(upsample_strides)} respectively"
            )
        if len(layer_nums) != len(num_upsample_filters):
            raise ValueError(
                f"Layer nums and num upsample strides must have same length, "
                f"got {len(layer_nums)}, {len(num_upsample_filters)} respectively"
            )

        # 看着没用
        factors = []
        for i in range(len(layer_nums)):
            factors.append(np.prod(layer_strides[: i + 1]) // upsample_strides[i])

        if use_norm:
            batch_norm2d_class = nn.BatchNorm2d
        else:
            batch_norm2d_class = ops.Identity()

        block2_input_filters = num_filters[0]

        if use_bev:
            self.bev_extractor = nn.SequentialCell(
                nn.Conv2d(6, 32, 3, padding=1, pad_mode="pad", has_bias=not use_norm),
                batch_norm2d_class(32, eps=1e-3, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1, pad_mode="pad", has_bias=not use_norm),
                batch_norm2d_class(64, eps=1e-3, momentum=0.99),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = nn.SequentialCell(
            nn.Conv2d(
                num_input_filters,
                num_filters[0],  # 64
                kernel_size=3,
                padding=1,
                pad_mode="pad",
                stride=layer_strides[0],
                has_bias=not use_norm,
            ),
            batch_norm2d_class(num_filters[0], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )

        for i in range(layer_nums[0]):
            self.block1.append(
                nn.Conv2d(
                    num_filters[0],
                    num_filters[0],
                    3,
                    padding=1,
                    pad_mode="pad",
                    has_bias=not use_norm,
                )
            )
            self.block1.append(batch_norm2d_class(num_filters[0]))
            self.block1.append(nn.ReLU())
        self.deconv1 = nn.SequentialCell(
            nn.Conv2dTranspose(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0],
                has_bias=not use_norm,
            ),
            batch_norm2d_class(num_upsample_filters[0], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        self.block2 = nn.SequentialCell(
            nn.Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                padding=1,
                pad_mode="pad",
                stride=layer_strides[1],
                has_bias=not use_norm,
            ),
            batch_norm2d_class(num_filters[1], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.append(
                nn.Conv2d(
                    num_filters[1],
                    num_filters[1],
                    3,
                    padding=1,
                    pad_mode="pad",
                    has_bias=not use_norm,
                )
            )
            self.block2.append(
                batch_norm2d_class(num_filters[1], eps=1e-3, momentum=0.99)
            )
            self.block2.append(nn.ReLU())
        self.deconv2 = nn.SequentialCell(
            nn.Conv2dTranspose(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1],
                has_bias=not use_norm,
            ),
            batch_norm2d_class(num_upsample_filters[1], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        self.block3 = nn.SequentialCell(
            nn.Conv2d(
                num_filters[1],
                num_filters[2],
                3,
                padding=1,
                pad_mode="pad",
                stride=layer_strides[2],
                has_bias=not use_norm,
            ),
            batch_norm2d_class(num_filters[2], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.append(
                nn.Conv2d(
                    num_filters[2],
                    num_filters[2],
                    3,
                    padding=1,
                    pad_mode="pad",
                    has_bias=not use_norm,
                )
            )
            self.block3.append(
                batch_norm2d_class(num_filters[2], eps=1e-3, momentum=0.99)
            )
            self.block3.append(nn.ReLU())
        self.deconv3 = nn.SequentialCell(
            nn.Conv2dTranspose(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2],
                has_bias=not use_norm,
            ),
            batch_norm2d_class(num_upsample_filters[2], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(
            sum(num_upsample_filters), num_cls, kernel_size=1, has_bias=True
        )
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters),
            num_anchor_per_loc * box_code_size,
            kernel_size=1,
            has_bias=True,
        )
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1, has_bias=True
            )

        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=1)

    def construct(self, x, anchors, bev=None):
        """forward graph"""
        x = self.block1(x.astype(mstype.float32))
        up1 = self.deconv1(x)
        if self.use_bev:
            bev[:, -1] = ops.Log()(1 + bev[:, -1]) / _log16()
            bev[:, -1] = ops.clip_by_value(
                bev[:, -1],
                clip_value_min=bev[:, -1].min(),
                clip_value_max=_create_on_value(),
            )
            x = self.concat([x, self.bev_extractor(bev)])
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = self.concat([up1, up2, up3])
        # the output shape = (batch_size, 384, 248, 216)", x.shape)
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        # [N, C, y(H), x(W)]
        box_preds = self.transpose(
            box_preds, (0, 2, 3, 1)
        )  # batch_size, y(H-248), x(W-216),C(18)
        cls_preds = self.transpose(cls_preds, (0, 2, 3, 1))
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = self.transpose(dir_cls_preds, (0, 2, 3, 1))

        if self.training:
            return box_preds, cls_preds, dir_cls_preds
        (
            batch_box_preds,
            batch_cls_preds,
            batch_dir_preds,
        ) = self.generate_predicted_boxes(
            box_preds, cls_preds, dir_cls_preds, anchors, self.box_coder
        )
        return batch_box_preds, batch_cls_preds, batch_dir_preds

    def generate_predicted_boxes(
        self, batch_box_preds, batch_cls_preds, batch_dir_preds, anchors, box_coder
    ):
        batch_size = anchors.shape[0]
        batch_anchors = anchors.view(batch_size, -1, 7)
        batch_box_preds = batch_box_preds.view(batch_size, -1, box_coder.code_size)
        num_class_with_bg = self.num_class
        if not self.encode_background_as_zeros:
            num_class_with_bg = self.num_class + 1
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = box_coder.decode(batch_box_preds, batch_anchors)

        if self.use_direction_classifier:
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
            if not self.use_self_train:
                dir_offset = 0.78539
                dir_limit_offset = 0.0
                dir_labels = ops.Argmax(axis=-1)(batch_dir_preds)
                period = 2 * np.pi / 2
                dir_rot = limit_period(
                    batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
                )
                batch_box_preds[..., 6] = (
                    dir_rot
                    + dir_offset
                    + period * dir_labels.astype(batch_box_preds.dtype)
                )
        else:
            batch_dir_preds = [None] * batch_size
        return batch_box_preds, batch_cls_preds, batch_dir_preds


class PointPillarsNet(nn.Cell):
    """PointPillars net"""

    def __init__(
        self,
        box_coder,
        output_shape,
        use_self_train=True,
        num_class=2,
        num_input_features=4,
        vfe_num_filters=(32, 128),
        with_distance=False,
        rpn_layer_nums=(3, 5, 5),
        rpn_layer_strides=(2, 2, 2),
        rpn_num_filters=(128, 128, 256),
        rpn_upsample_strides=(1, 2, 4),
        rpn_num_upsample_filters=(256, 256, 256),
        use_norm=True,
        use_direction_classifier=True,
        encode_background_as_zeros=True,
        num_anchor_per_loc=2,
        code_size=7,
        use_bev=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        use_sigmoid_score=True,
        pre_max_size=900,
    ):
        super().__init__()

        self.num_class = num_class
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_direction_classifier = use_direction_classifier
        self.use_bev = use_bev
        self.code_size = code_size
        self.num_anchor_per_loc = num_anchor_per_loc
        self.box_coder = box_coder
        self.use_sigmoid_score = use_sigmoid_score
        self.pre_max_size = pre_max_size

        self.voxel_feature_extractor = PillarFeatureNet(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=voxel_size,
            pc_range=pc_range,
        )
        self.middle_feature_extractor = PointPillarsScatter(
            output_shape=output_shape, num_input_features=vfe_num_filters[-1]
        )
        num_rpn_input_filters = self.middle_feature_extractor.n_channels

        self.rpn = RPN(
            box_coder=box_coder,
            use_norm=True,
            use_self_train=use_self_train,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=num_rpn_input_filters,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_bev=use_bev,
            box_code_size=code_size,
        )
        self.select = ops.Select()
        self.zeroslike = ops.ZerosLike()

    def construct(self, voxels, num_points, coors, anchors, anchors_mask, bev_map=None):
        """forward graph"""
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        if self.use_bev:
            preds = self.rpn(spatial_features, anchors, bev_map)
        else:
            preds = self.rpn(spatial_features, anchors)
        if self.training:
            return preds
        selected_data_all = self.post_processing(preds, anchors_mask)
        return selected_data_all

    def get_total_scores(self, cls_preds):
        """get total scores"""
        if self.encode_background_as_zeros:
            total_scores = ops.sigmoid(cls_preds)
        else:
            # encode background as first element in one-hot vector
            if self.use_sigmoid_score:
                total_scores = ops.sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = ops.softmax(cls_preds, axis=-1)[..., 1:]
        return total_scores

    def get_selected_data(
        self, total_scores, box_preds, dir_labels, num_class_with_bg, mask
    ):
        """get selected data"""
        if num_class_with_bg == 1:
            top_scores = total_scores.squeeze(-1)
            top_labels = ops.zeros(total_scores.shape[0], mstype.int64)
        else:
            top_scores, top_labels = ops.max(total_scores, axis=-1)
            # top_labels, top_scores = ops.ArgMaxWithValue(axis=-1)(total_scores)
        top_scores = ops.select(mask, top_scores, self.zeroslike(top_scores) - 1)
        scores, indices = ops.top_k(top_scores, self.pre_max_size)
        return box_preds, top_labels, scores, indices, dir_labels

    def post_processing(self, preds, anchors_mask):
        batch_box_preds, batch_cls_preds, batch_dir_preds = preds
        batch_size = batch_box_preds.shape[0]
        num_class_with_bg = self.num_class
        if not self.encode_background_as_zeros:
            num_class_with_bg = self.num_class + 1
        selected_data_all = []

        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            cls_preds = batch_cls_preds[i]
            dir_preds = batch_dir_preds[i]
            a_mask = anchors_mask[i]
            # mask = ops.greater(a_mask, 0)
            mask = a_mask > 0
            dir_labels = None
            if self.use_direction_classifier:
                dir_labels = ops.argmax(dir_preds, dim=-1)
            total_scores = self.get_total_scores(cls_preds)
            selected_data = self.get_selected_data(
                total_scores, box_preds, dir_labels, num_class_with_bg, mask
            )
            selected_data_all.append(selected_data)
        return selected_data_all


class PointPillarsWithLossCell(nn.Cell):
    """PointPillars with loss cell"""

    def __init__(self, network, cfg):
        super().__init__()
        self.network = network
        self.cfg = cfg
        loss_cfg = cfg["loss"]
        self.loss_cls = SigmoidFocalClassificationLoss(
            gamma=loss_cfg["classification_loss"]["gamma"],
            alpha=loss_cfg["classification_loss"]["alpha"],
        )
        self.loss_loc = WeightedSmoothL1LocalizationLoss(
            sigma=loss_cfg["localization_loss"]["sigma"],
            code_weights=loss_cfg["localization_loss"]["code_weight"],
        )
        self.loss_dir = WeightedSoftmaxClassificationLoss()
        self.w_cls_loss = loss_cfg["classification_weight"]
        self.w_loc_loss = loss_cfg["localization_weight"]
        self.w_dir_loss = cfg["direction_loss_weight"]
        self._pos_cls_weight = cfg["pos_class_weight"]
        self._neg_cls_weight = cfg["neg_class_weight"]
        self.code_size = network.code_size

    def construct(self, *args):
        """forward graph"""
        voxels, num_points, coors, bev_map, labels, reg_targets, anchors = args
        batch_size_dev = labels.shape[0]
        preds = self.network(voxels, num_points, coors, anchors, bev_map)
        if self.cfg["use_direction_classifier"]:
            box_preds, cls_preds, dir_cls_preds = preds
            dir_targets = get_direction_target(anchors, reg_targets)
            dir_logits = dir_cls_preds.view(batch_size_dev, -1, 2)
            weights = (labels > 0).astype(dir_logits.dtype)
            weights /= ops.clip_by_value(
                weights.sum(-1, keepdims=True),
                clip_value_min=_create_on_value(),
                clip_value_max=weights.sum(-1, keepdims=True)
                .astype(mstype.float32)
                .max(),
            )
            dir_loss = self.loss_dir(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev
            loss = dir_loss * self.w_dir_loss
        else:
            loss = 0
            box_preds, cls_preds = preds

        cls_weights, reg_weights, cared = prepare_loss_weights(
            labels,
            pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight,
            dtype=voxels.dtype,
        )
        cls_targets = labels * cared.astype(labels.dtype)
        cls_targets = ops.ExpandDims()(cls_targets, -1)

        loc_loss, cls_loss = create_loss(
            self.loss_loc,
            self.loss_cls,
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights,
            reg_targets=reg_targets,
            reg_weights=reg_weights,
            num_class=self.cfg["num_class"],
            encode_rad_error_by_sin=self.cfg["encode_rad_error_by_sin"],
            encode_background_as_zeros=self.cfg["encode_background_as_zeros"],
            box_code_size=self.code_size,
        )
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self.w_loc_loss
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self._pos_cls_weight
        cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self.w_cls_loss
        loss += loc_loss_reduced + cls_loss_reduced
        return loss


class TrainingWrapper(nn.Cell):
    """training wrapper"""

    def __init__(self, network, optimizer, sens=1.0):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
            ParallelMode.DATA_PARALLEL,
            ParallelMode.HYBRID_PARALLEL,
        ]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(
                optimizer.parameters, mean, degree
            )

    def construct(self, *args):
        """forward graph"""
        loss = self.network(*args)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, self.weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        t = ops.depend(loss, self.optimizer(grads))
        return t
