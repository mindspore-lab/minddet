"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import mindspore.nn as nn
import mindspore.numpy as mnp
import numpy as np
from det3d_ms.models.readers.custom_bn import BatchNorm2dMasked
from det3d_ms.models.utils import get_paddings_indicator
from mindspore import Tensor, ops
from mindspore.common import dtype as mstype

from ..registry import BACKBONES, READERS


class PFNLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.99)
        self.norm_cfg = norm_cfg

        self.linear = nn.Dense(in_channels, self.units, has_bias=False).to_float(
            mstype.float32
        )
        self.norm = BatchNorm2dMasked(self.units, eps=1e-3, momentum=0.99)
        self.relu = nn.ReLU()
        self.transpose = ops.Transpose()
        self.expand_dims = ops.ExpandDims()
        self.concat = ops.Concat(axis=2)

    def construct(self, inputs, mask):
        """
        inputs: 0: (60000, 20, 10), 1: (60000, 20, 64)
        mask: 0: (60000,), 1: (60000,)
        """
        x = self.linear(inputs)
        # x = ops.Cast()(x, mstype.float32)
        # x = self.transpose(self.norm(self.transpose(x, (0, 2, 1).reshape())), (0, 2, 1))
        # x = self.transpose(self.norm(self.expand_dims(self.transpose(x, (0, 2, 1)), -1), mask).squeeze(-1), (0,2,1))
        x = self.norm(x, mask)
        x = self.relu(x)

        x_max = ops.ReduceMax()(x, 1)
        if self.last_vfe:
            return x_max
        else:
            x_repeat = mnp.tile(self.expand_dims(x_max, 1), (1, inputs.shape[1], 1))
            x_concatenated = self.concat([x, x_repeat])
            return x_concatenated


@READERS.register_module
class PillarFeatureNet(nn.Cell):
    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
        virtual=False,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.CellList(pfn_layers)

        self.virtual = virtual

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.concat = ops.Concat(axis=-1)
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()

    def construct(self, features, num_voxels, coors):
        """
        :param features: shape (60000, 20, 5), float32
        :param num_voxels: shape (60000,), int32
        :param coors: shape (60000, 4), int32
        :return:
        """
        if self.virtual:
            virtual_point_mask = features[..., -2] == -1
            virtual_points = features[virtual_point_mask]
            virtual_points[..., -2] = 1
            features[..., -2] = 0
            features[virtual_point_mask] = virtual_points

        data_type = features.dtype
        # Find distance of x, y, and z from cluster center
        # features = features[:, :, :self.num_input]
        num_voxels_for_div = num_voxels.copy()
        num_voxels_for_div = ops.select(
            num_voxels_for_div > 0,
            num_voxels_for_div,
            Tensor(np.ones(num_voxels.shape), dtype=num_voxels_for_div.dtype),
        )
        points_mean = features[:, :, :3].sum(axis=1, keepdims=True) / self.cast(
            num_voxels_for_div, data_type
        ).view((-1, 1, 1))
        # dynamic shape
        mask = self.cast(num_voxels > 0, mstype.int32)
        mask_points = mnp.tile(mask.reshape(-1, 1, 1), (1,) + points_mean.shape[1:])
        points_mean = mnp.where(mask_points, points_mean, ops.ZerosLike()(points_mean))

        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center_0 = features[:, :, 0] - (
            self.expand_dims(self.cast(coors[:, 3], data_type), 1) * self.vx
            + self.x_offset
        )
        f_center_1 = features[:, :, 1] - (
            self.expand_dims(self.cast(coors[:, 2], data_type), 1) * self.vy
            + self.y_offset
        )
        f_center = ops.Concat(axis=2)(
            (f_center_0.expand_dims(-1), f_center_1.expand_dims(-1))
        )
        # dynamic shape
        mask_center = mnp.tile(mask.reshape(-1, 1, 1), (1,) + f_center.shape[1:])
        f_center = mnp.where(mask_center, f_center, ops.ZerosLike()(f_center))

        # Combine feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = ops.norm(features[:, :, :3], dim=2)
            features_ls.append(points_dist)
        features = self.concat(features_ls)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask_voxels = get_paddings_indicator(num_voxels, voxel_count, axis=0)  # 0.0009s
        mask_voxels = self.cast(self.expand_dims(mask_voxels, -1), data_type)

        features *= mask_voxels

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features, mask)

        return features.squeeze()  # 0.0


@BACKBONES.register_module
class PointPillarsScatter(nn.Cell):
    def __init__(
        self, num_input_features=64, norm_cfg=None, name="PointPillarsScatter", **kwargs
    ):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

    def construct(self, voxel_features, coords, batch_size, input_shape):
        batch_canvas = ops.transpose(
            ops.ScatterNd()(
                ops.Stack(-1)([coords[..., 0], coords[..., 2], coords[..., 3]]),
                voxel_features,
                (batch_size, 512, 512, self.nchannels),
            ),
            (0, 3, 1, 2),
        )
        return batch_canvas


if __name__ == "__main__":
    dtype = mstype.float32
    batch_size = 1002
    pnf_layer = PillarFeatureNet(5, num_filters=(64, 64))
    print(
        pnf_layer(
            Tensor(np.zeros([batch_size, 20, 5]), dtype=mstype.float32),
            Tensor(np.zeros([batch_size]), dtype=mstype.float32),
            Tensor(np.zeros([batch_size, 4]), dtype=mstype.float32),
        ).shape
        == (batch_size, 64)
    )
    pointpillars = PointPillarsScatter()
    batch_size1 = 4
    print(
        pointpillars(
            Tensor(np.zeros([batch_size, 64]), dtype=mstype.float32),
            Tensor(np.zeros([batch_size, 4]), dtype=mstype.int32),
            batch_size1,
            input_shape=(512, 512),
        ).shape
        == (batch_size1, 64, 512, 512)
    )
