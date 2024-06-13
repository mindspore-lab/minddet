"""
Decode from heads for evaluation
"""

import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from .utils import GatherFeature, TransposeGatherFeature


class NMS(nn.Cell):
    """
    Non-maximum suppression

    Args:
        kernel(int): Maxpooling kernel size. Default: 3.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: False.

    Returns:
        Tensor, heatmap after non-maximum suppression.
    """

    def __init__(self, kernel=3, enable_nms_fp16=False):
        super(NMS, self).__init__()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.equal = ops.Equal()
        self.Abs = P.Abs()

        # self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        # self.max_pool_ = nn.MaxPool2d(kernel_size=3, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel, stride=1, pad_mode="same")

        self.enable_fp16 = enable_nms_fp16
        self.print = ops.Print()

    def construct(self, heat):
        """Non-maximum suppression"""
        dtype = self.dtype(heat)
        if self.enable_fp16:
            heat = self.cast(heat, mstype.float16)
            heat_max = self.max_pool_(heat)
            keep = self.equal(heat, heat_max)
            keep = self.cast(keep, dtype)
            heat = self.cast(heat, dtype)
        else:
            # heat_pad = self.pad(heat)
            # heat_max = self.max_pool_(heat_pad.astype(mstype.float16)).astype(heat.dtype)
            # error = self.cast((heat - heat_max), mstype.float32)
            # abs_error = self.Abs(error)
            # abs_out = self.Abs(heat)
            # error = abs_error / (abs_out + 1e-12)
            # keep = P.Select()(P.LessEqual()(error, 1e-3),
            #                   P.Fill()(ms.float32, P.Shape()(error), 1.0),
            #                   P.Fill()(ms.float32, P.Shape()(error), 0.0))
            dtype = self.dtype(heat)
            hmax = self.max_pool(heat)
            keep = self.equal(heat, hmax)
            keep = self.cast(keep, dtype)
        heat = heat * keep
        return heat


class GatherTopK(nn.Cell):
    """
    Gather topk features through all channels

    Args: None

    Returns:
        Tuple of Tensors, top_k scores, indexes, category ids, and the indexes in height and width direcction.
    """

    def __init__(self):
        super(GatherTopK, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.topk = ops.TopK(sorted=True)
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.gather_feat = GatherFeature()
        # The ops.Mod() operator will produce errors on the Ascend 310
        self.mod = P.FloorMod()
        # self.div = ops.div(rounding_mode='trunc')
        self.div = ops.Div()

    def construct(self, scores, K=100):
        """gather top_k"""
        b, c, _, w = self.shape(scores)
        scores = self.reshape(scores, (b, c, -1))
        # (b, c, K)

        topk_scores, topk_inds = self.topk(scores, 100)

        topk_ys = self.div(topk_inds, w)
        topk_xs = self.mod(topk_inds, w)
        # (b, K)
        topk_score, topk_ind = self.topk(self.reshape(topk_scores, (b, -1)), 100)
        topk_clses = self.cast(self.div(topk_ind, 100), self.dtype(scores))
        topk_inds = self.gather_feat(self.reshape(topk_inds, (b, -1, 1)), topk_ind)
        topk_inds = self.reshape(topk_inds, (b, 100))
        topk_ys = self.gather_feat(self.reshape(topk_ys, (b, -1, 1)), topk_ind)
        topk_ys = self.cast(self.reshape(topk_ys, (b, 100)), self.dtype(scores))
        topk_xs = self.gather_feat(self.reshape(topk_xs, (b, -1, 1)), topk_ind)
        topk_xs = self.cast(self.reshape(topk_xs, (b, 100)), self.dtype(scores))
        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    # def construct(self, scores, k=100):
    #     b, c, h, w = self.shape(scores)
    #     scores = self.reshape(scores, (b, -1))
    #     topk_scores, topk_inds = self.topk(scores, k)
    #     topk_clses = ops.div(topk_inds, h * w, rounding_mode='trunc')
    #     # topk_clses = self.cast(topk_clses, self.dtype(topk_scores))
    #     topk_inds = self.mod(topk_inds, h * w)
    #     topk_ys = ops.div(topk_inds, w, rounding_mode='trunc')
    #     topk_xs = self.mod(topk_inds, w)
    #     return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


class DetectionDecode(nn.Cell):
    """
    Decode from heads to gather multi-objects info.

    Args:
        net_config(edict): config info for CenterNet network.
        K(int): maximum objects number. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: True.

    Returns:
        Tensor, multi-objects detections.
    """

    def __init__(self, net_config, K=100, enable_nms_fp16=False):
        super(DetectionDecode, self).__init__()
        self.nms = NMS(enable_nms_fp16=enable_nms_fp16)
        self.shape = ops.Shape()
        self.gather_topk = GatherTopK()
        self.half = ops.Split(axis=-1, output_num=2)
        self.add = ops.Add()
        self.concat_a2 = ops.Concat(axis=2)
        self.trans_gather_feature = TransposeGatherFeature()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.reg_offset = net_config.reg_offset
        self.Sigmoid = nn.Sigmoid()
        self.print = ops.Print()

    def construct(self, feature, K=100):
        """gather detections"""
        heat = feature["hm"]
        b, _, _, _ = self.shape(heat)
        heat = self.nms(heat)
        # self.print(heat)
        scores, inds, clses, ys, xs = self.gather_topk(heat)
        # self.print(scores)
        # self.print(inds)
        # self.print(clses)
        # self.print(ys)
        # self.print(xs)
        ys = self.reshape(ys, (b, K, 1))
        xs = self.reshape(xs, (b, K, 1))

        wh = feature["wh"]
        wh = self.trans_gather_feature(wh, inds)
        # wh = self.reshape(wh, (b, K, 2))

        ws, hs = self.half(wh)
        # ws = wh[..., 0:1]
        # hs = wh[..., 1:2]
        if self.reg_offset:
            reg = feature["reg"]
            reg = self.trans_gather_feature(reg, inds)
            reg = self.reshape(reg, (b, K, 2))
            reg_w, reg_h = self.half(reg)
            # reg_w = reg[:, :, 0:1]
            # reg_h = reg[:, :, 1:2]
            ys = self.add(ys, reg_h)
            xs = self.add(xs, reg_w)
        else:
            ys = ys + 0.5
            xs = xs + 0.5

        c1 = Tensor(xs - ws / 2, mstype.float32)
        c2 = Tensor(ys - hs / 2, mstype.float32)
        c3 = Tensor(xs + ws / 2, mstype.float32)
        c4 = Tensor(ys + hs / 2, mstype.float32)
        bboxes = self.concat_a2((c1, c2, c3, c4))

        clses = self.expand_dims(clses, 2)
        scores = self.expand_dims(scores, 2)
        # clses = self.reshape(clses, (b, K, 1))
        # scores = self.reshape(scores, (b, K, 1))
        detection = self.concat_a2((bboxes, scores, clses))
        return detection
