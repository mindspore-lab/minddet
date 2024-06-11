# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------

import copy
import logging

import mindspore.numpy as mnp
import numpy as np
from det3d_ms.models.losses.centernet_loss import FastFocalLoss, RegLoss
from det3d_ms.ops.nms_cpu import NMS
from mindspore import Tensor, context, nn, ops
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Constant
from mindspore.ops import constexpr

from ..registry import HEADS


@constexpr
def generate_tensor(data, dtype):
    return Tensor(data, dtype)


class SepHead(nn.Cell):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
    ):
        super(SepHead, self).__init__()

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = []
            for i in range(num_conv - 1):
                fc.append(
                    nn.Conv2d(
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        has_bias=True,
                    )
                )
                if bn:
                    fc.append(nn.BatchNorm2d(head_conv))
                    # fc.append(nn.SyncBatchNorm(head_conv))
                fc.append(nn.ReLU())
            if "hm" in head:
                fc.append(
                    nn.Conv2d(
                        head_conv,
                        classes,
                        kernel_size=final_kernel,
                        stride=1,
                        has_bias=True,
                        bias_init=Constant(init_bias),
                    )
                )
            else:
                fc.append(
                    nn.Conv2d(
                        head_conv,
                        classes,
                        kernel_size=final_kernel,
                        stride=1,
                        has_bias=True,
                    )
                )  # , weight_init='xavier_uniform'))

            fc = nn.SequentialCell(fc)
            # if 'hm' in head:
            #    fc[-1].bias.data.fill_(init_bias)
            # else:
            #    for m in fc.modules():
            #        if isinstance(m, nn.Conv2d):
            #            kaiming_init(m)

            self.__setattr__(head, fc)

    def construct(self, x):
        return {
            "reg": self.reg(x),
            "height": self.height(x),
            "dim": self.dim(x),
            "rot": self.rot(x),
            "vel": self.vel(x),
            "hm": self.hm(x),
        }


@HEADS.register_module
class CenterHead(nn.Cell):
    def __init__(
        self,
        in_channels=[
            128,
        ],
        tasks=[],
        dataset="nuscenes",
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.box_n_dim = 9 if "vel" in common_heads else 7
        self.use_direction_classifier = False

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(f"num_classes: {num_classes}")

        # a shared convolution
        self.shared_conv = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels, share_conv_channel, kernel_size=3, has_bias=True
                ),
                nn.BatchNorm2d(share_conv_channel),
                # nn.SyncBatchNorm(share_conv_channel),
                nn.ReLU(),
            ]
        )

        self.tasks = []

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(
                    share_conv_channel,
                    heads,
                    bn=True,
                    init_bias=init_bias,
                    final_kernel=3,
                )
            )
        self.tasks = nn.CellList(self.tasks)
        logger.info("Finish CenterHead Initialization")
        self.transpose = ops.Transpose()
        self.concat1 = ops.Concat(axis=1)
        self.concat2 = ops.Concat(axis=2)
        self.cast = ops.Cast()
        self.logical_and = ops.LogicalAnd()
        self.sigmoid = ops.Sigmoid()
        self.exp = ops.Exp()
        self.meshgrid = ops.Meshgrid(indexing="ij")
        self.atan2 = ops.Atan2()
        self.argmax_with_value = ops.ArgMaxWithValue(axis=-1)
        self.concat = ops.Concat()
        self.select = ops.Select()
        self.zeroslike = ops.ZerosLike()
        self.device = context.get_context("device_target")
        self.nms_type = "gpu"
        if self.device == "Ascend":
            self.nms_type = "cpu"
        self.nms = NMS()
        self.topK = ops.TopK()
        self.minimum = ops.Minimum()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts, x

    def _sigmoid(self, x):
        y = ops.clip_by_value(
            self.sigmoid(x), clip_value_min=1e-4, clip_value_max=1 - 1e-4
        )
        return y

    def loss(self, example, preds_dicts, test_cfg):
        loss_rets = [1, 2, 3, 4, 5, 6]
        hm_loss_rets = [1, 2, 3, 4, 5, 6]
        loc_loss_rets = [1, 2, 3, 4, 5, 6]
        loc_loss_elem_rets = [1, 2, 3, 4, 5, 6]
        num_positive_rets = [1, 2, 3, 4, 5, 6]
        total_loss = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict["hm"] = self._sigmoid(preds_dict["hm"])
            hm_loss = self.crit(
                preds_dict["hm"],
                example["hm"][:, task_id, : self.num_classes[task_id]],
                example["ind"][:, task_id],
                example["mask"][:, task_id],
                example["cat"][:, task_id],
            )

            target_box = example["anno_box"][:, task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset in ["waymo", "nuscenes"]:
                if "vel" in preds_dict:
                    preds_dict["anno_box"] = self.concat1(
                        (
                            preds_dict["reg"],
                            preds_dict["height"],
                            preds_dict["dim"],
                            preds_dict["vel"],
                            preds_dict["rot"],
                        )
                    )
                else:
                    preds_dict["anno_box"] = self.concat1(
                        (
                            preds_dict["reg"],
                            preds_dict["height"],
                            preds_dict["dim"],
                            preds_dict["rot"],
                        )
                    )
                    target_box = target_box[
                        ..., [0, 1, 2, 3, 4, 5, -2, -1]
                    ]  # remove vel target

            # Regression loss for dimension, offset, height, rotation
            box_loss = self.crit_reg(
                preds_dict["anno_box"],
                example["mask"][:, task_id],
                example["ind"][:, task_id],
                target_box,
            )

            loc_loss = (box_loss * self.code_weights).sum()

            loss = hm_loss + self.weight * loc_loss
            loss_rets[task_id] = loss
            hm_loss_rets[task_id] = hm_loss
            loc_loss_rets[task_id] = loc_loss
            loc_loss_elem_rets[task_id] = box_loss
            num_positive_rets[task_id] = self.cast(
                example["mask"][:, task_id], mstype.float32
            ).sum()
            total_loss = total_loss + loss
        return total_loss

    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionally supports double flip testing"""
        # get loss info
        rets = []
        metas = []

        post_center_range = generate_tensor(
            test_cfg["post_center_limit_range"], preds_dicts[0]["hm"].dtype
        )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C
            for key in preds_dict.keys():
                preds_dict[key] = self.transpose(
                    preds_dict[key], (0, 2, 3, 1)
                )  # val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict["hm"].shape[0]

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]

            batch_hm = self.sigmoid(preds_dict["hm"])

            batch_dim = self.exp(preds_dict["dim"])

            batch_rots = preds_dict["rot"][..., 0:1]
            batch_rotc = preds_dict["rot"][..., 1:2]
            batch_reg = preds_dict["reg"]
            batch_hei = preds_dict["height"]

            batch_rot = self.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.shape

            batch_reg = batch_reg.reshape(batch, H * W, 2)
            batch_hei = batch_hei.reshape(batch, H * W, 1)

            batch_rot = batch_rot.reshape(batch, H * W, 1)
            batch_dim = batch_dim.reshape(batch, H * W, 3)
            batch_hm = batch_hm.reshape(batch, H * W, num_cls)

            ys, xs = self.meshgrid((mnp.arange(0, H), mnp.arange(0, W)))
            ys = self.cast(
                mnp.tile(ys.view((1, H, W)), (batch, 1, 1)), batch_hm.dtype
            )  # .repeat(batch, 1, 1)#.to(batch_hm)
            xs = self.cast(
                mnp.tile(xs.view((1, H, W)), (batch, 1, 1)), batch_hm.dtype
            )  # .repeat(batch, 1, 1)#.to(batch_hm)

            xs = xs.view((batch, -1, 1)) + batch_reg[:, :, 0:1]
            ys = ys.view((batch, -1, 1)) + batch_reg[:, :, 1:2]
            xs = (
                xs * test_cfg["out_size_factor"] * test_cfg["voxel_size"][0]
                + test_cfg["pc_range"][0]
            )
            ys = (
                ys * test_cfg["out_size_factor"] * test_cfg["voxel_size"][1]
                + test_cfg["pc_range"][1]
            )

            if "vel" in preds_dict:
                batch_vel = preds_dict["vel"]
                batch_vel = batch_vel.reshape(batch, H * W, 2)
                batch_box_preds = self.concat2(
                    [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot]
                )
            else:
                batch_box_preds = self.concat2(
                    [xs, ys, batch_hei, batch_dim, batch_rot]
                )

            metas.append(meta_list)

            # before post-processing time consumption
            # 0 0.0035s
            # 1 0.0037s
            # 2 0.0035s
            # 3 0.0034s
            # 4 0.0034s
            # 5 0.0033s

            rets.append(
                self.post_processing(
                    batch_box_preds, batch_hm, test_cfg, post_center_range, task_id
                )
            )
            # post-pprocessing time consumption
            # 0 0.03s
            # 1 0.009s
            # 2 0.008s
            # 3 0.008s
            # 4 0.001s
            # 5 0.0013s

        # Merge branches results
        # ret_list = []
        # num_samples = len(rets[0])
        # ret_list = []
        # for i in range(num_samples):
        #     ret = {'metadata':metas[0][i]}
        #     for k in rets[0][i].keys():
        #         if k in ["box3d_lidar", "scores"]:
        #             data = []
        #             for item in rets:
        #                 if (item[i][k].shape[0] > 0):
        #                     data.append(item[i][k])
        #             if len(data) > 0:
        #                 ret[k] = self.concat(data)
        #         elif k in ["label_preds"]:
        #             flag = 0
        #             data = []
        #             for j, num_class in enumerate(self.num_classes):
        #                 if rets[j][i][k].shape[0] > 0:
        #                     data.append(rets[j][i][k] + flag)
        #                 flag += num_class
        #             if len(data) > 0:
        #                 ret[k] = self.concat(data)

        #     ret_list.append(ret)

        return rets, example["token"]

    def post_processing(
        self, batch_box_preds, batch_hm, test_cfg, post_center_range, task_id
    ):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            labels, scores = self.argmax_with_value(hm_preds)

            score_mask = scores > test_cfg["score_threshold"]
            distance_mask = self.logical_and(
                (box_preds[..., :3] >= post_center_range[:3]).all(1),
                (box_preds[..., :3] <= post_center_range[3:]).all(1),
            )

            mask = self.logical_and(distance_mask, score_mask)
            box_preds_shape = box_preds.shape
            box_preds = self.cast(box_preds, mstype.float32)
            scores = self.cast(scores, mstype.float32)
            labels = self.cast(labels, mstype.int32)

            scores = self.select(mask, scores, self.zeroslike(scores) - 1)
            labels = self.select(mask, labels, self.zeroslike(labels) - 1)
            box_preds = self.select(
                mnp.tile(mask.reshape(-1, 1), (1, box_preds_shape[1])).reshape(-1),
                box_preds.reshape(-1),
                self.zeroslike(box_preds).reshape(-1),
            ).reshape(-1, box_preds_shape[1])
            box_preds[:, -1] = -box_preds[:, -1] - np.pi / 2

            if self.nms_type == "npu":
                boxes_for_nms = box_preds[:, [0, 1, 2, 4, 3, 5, -1, 6]]
            else:
                boxes_for_nms = box_preds[:, [0, 1, 2, 4, 3, 5, -1]]
            order = self.topK(scores, test_cfg["nms"]["nms_pre_max_size"])[1]
            scores_sorted = scores[order]
            boxes_for_nms_sorted = boxes_for_nms[order]
            boxes_sorted = box_preds[order]
            labels_sorted = labels[order]
            mask_num = self.cast(
                self.cast(mask, mstype.float32)[order].sum(), mstype.int32
            )
            keep, num_out = self.nms(
                boxes_for_nms_sorted, test_cfg["nms"]["nms_iou_threshold"]
            )  # 0.003s
            boxes_sorted[:, -1] = -boxes_sorted[:, -1] - np.pi / 2
            selected_scores = scores_sorted[keep]
            selected_boxes = boxes_sorted[keep]
            selected_labels = labels_sorted[keep]

            prediction_dict = [
                selected_boxes,
                selected_scores,
                selected_labels,
                self.minimum(
                    self.minimum(num_out, mask_num),
                    test_cfg["nms"]["nms_post_max_size"],
                ),
            ]

            prediction_dicts.append(prediction_dict)

        return prediction_dicts


if __name__ == "__main__":
    # context.set_context(mode=context.GRAPH_MODE, device_target="Ascend") # GRAPH_MODE
    tasks = [
        dict(num_class=1, class_names=["car"]),
        dict(num_class=2, class_names=["truck", "construction_vehicle"]),
        dict(num_class=2, class_names=["bus", "trailer"]),
        dict(num_class=1, class_names=["barrier"]),
        dict(num_class=2, class_names=["motorcycle", "bicycle"]),
        dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
    ]
    model = CenterHead(
        in_channels=384,
        tasks=tasks,
        dataset="nuscenes",
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={
            "reg": (2, 2),
            "height": (1, 2),
            "dim": (3, 2),
            "rot": (2, 2),
            "vel": (2, 2),
        },
    ).to_float(mstype.float16)
    ret_dicts, x = model(Tensor(np.zeros([4, 384, 128, 128]), dtype=mstype.float16))
    print(x.shape == (4, 64, 128, 128))
    example = {}
    from pathlib import PosixPath

    example["metadata"] = [
        {
            "image_prefix": PosixPath("data/nuScenes"),
            "num_point_features": 5,
            "token": "b7f64f73e8a548488e6d85d9b0e13242",
        },
        {
            "image_prefix": PosixPath("data/nuScenes"),
            "num_point_features": 5,
            "token": "37535cd306954669a2b21578573b93e3",
        },
        {
            "image_prefix": PosixPath("data/nuScenes"),
            "num_point_features": 5,
            "token": "c5f58c19249d4137ae063b0e9ecd8b8e",
        },
        {
            "image_prefix": PosixPath("data/nuScenes"),
            "num_point_features": 5,
            "token": "3ffc4360d1084e6eae5067e87d79503f",
        },
    ]
    test_cfg = {
        "post_center_limit_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        "max_per_img": 500,
        "min_radius": [4, 12, 10, 1, 0.85, 0.175],
        "circular_nms": True,
        "nms": {
            "nms_pre_max_size": 1000,
            "nms_post_max_size": 83,
            "nms_iou_threshold": 0.2,
        },
        "score_threshold": 0.1,
        "pc_range": [-51.2, -51.2],
        "out_size_factor": 4,
        "voxel_size": [0.2, 0.2],
    }
    from easydict import EasyDict

    model.predict(example, ret_dicts, EasyDict(test_cfg))
    # model.loss(example, ret_dicts, EasyDict(test_cfg))
