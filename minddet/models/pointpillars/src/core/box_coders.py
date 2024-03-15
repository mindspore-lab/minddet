"""box coders"""
import time as time

import numpy as np
from mindspore import numpy as mnp
from mindspore import ops
from mindspore.profiler import Profiler
from src.core import box_np_ops, box_ops


class GroundBox3dCoder:
    """ground box 3d coder"""

    def __init__(self, linear_dim=False, vec_encode=False):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode

    def encode(self, inp, anchors):
        """encode"""
        return box_np_ops.second_box_encode(
            inp, anchors, self.vec_encode, self.linear_dim
        )

    def decode(self, inp, anchors):
        """decode"""
        batch_box_preds = box_ops.second_box_decode(inp, anchors, self.vec_encode)
        return batch_box_preds

    @property
    def code_size(self):
        """code size"""
        return 8 if self.vec_encode else 7


class BevBoxCoder:
    """bev box coder"""

    def __init__(self, linear_dim=False, vec_encode=False, z_fixed=-1.0, h_fixed=2.0):
        super().__init__()
        self.linear_dim = linear_dim
        self.z_fixed = z_fixed
        self.h_fixed = h_fixed
        self.vec_encode = vec_encode

    def encode(self, inp, anchors):
        """encode"""
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        inp = inp[..., [0, 1, 3, 4, 6]]
        return box_np_ops.bev_box_encode(inp, anchors, self.vec_encode, self.linear_dim)

    def decode(self, inp, anchors):
        """decode"""
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        ret = box_ops.bev_box_decode(inp, anchors, self.vec_encode, self.linear_dim)
        z_fixed = mnp.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype)
        h_fixed = mnp.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype)
        return ops.Concat(axis=-1)(
            [ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]]
        )

    @property
    def code_size(self):
        """code size"""
        return 6 if self.vec_encode else 5
