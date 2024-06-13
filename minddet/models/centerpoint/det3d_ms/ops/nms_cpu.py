import mindspore
import mindspore.ops as P
import numpy as np
from mindspore import Tensor, nn


class NMS(nn.Cell):
    def __init__(self):
        super(NMS, self).__init__()
        self.nms = P.Custom(
            "./det3d_ms/ops/nms_fast.so:boxes_iou_nms_cpu",
            out_shape=lambda x, _: (
                [
                    x[0],
                ],
                [
                    1,
                ],
            ),
            out_dtype=lambda x, _: (mindspore.int32, mindspore.int32),
            func_type="aot",
        )
        self.nms.add_prim_attr("primitive_target", "CPU")

    def construct(self, boxes, thresh):
        ret = self.nms(boxes, thresh)
        return ret[0], ret[1][0]


if __name__ == "__main__":
    nms = NMS()
    nms(Tensor(np.zeros((100, 7)), mindspore.float32), 0.1)
