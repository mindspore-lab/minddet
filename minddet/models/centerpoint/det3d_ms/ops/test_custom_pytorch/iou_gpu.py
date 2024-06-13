import os

import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import export
from mindspore.nn import Cell


# boxes_iou
# in1:  boxesa [M, 7] float32,  in2: boxesb [N, 7] float32
# out1: ans_iou [M,N] float32
class BoxesIouBevGpu(Cell):
    def __init__(self):
        super().__init__()
        self.boxes_iou = ops.Custom(
            "./iou_nms.so:BoxesIouBevGpu",
            out_shape=lambda a, b: [a[0], b[0]],
            out_dtype=lambda a, b: a,
            func_type="aot",
        )

    def construct(self, a, b):
        return self.boxes_iou(a, b)


# boxes_overlap
# in1:  boxesa [M, 7] float32,  in2: boxesb [N, 7] float32
# out1: ans_iou [M,N] float32
class BoxesOverlapBevGpu(Cell):
    def __init__(self):
        super(BoxesOverlapBevGpu, self).__init__()
        self.boxes_overlap = ops.Custom(
            "./iou_nms.so:BoxesOverlapBevGpu",
            out_shape=lambda a, b: [a[0], b[0]],
            out_dtype=lambda a, b: a,
            func_type="aot",
        )

    def construct(self, a, b):
        return self.boxes_overlap(a, b)


# nmsgpu
# in1:  boox [N, 7] float tensor,  in2: thresh [1], float tensor
# out1: keep [N]    long tensor,   out2: num_to_keep [1], int tensor
class NumGpu(Cell):
    def __init__(self):
        super(NumGpu, self).__init__()
        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        iou_path = father_path + "/iou_nms.so:NmsGpu"
        print("================", iou_path)
        self.nms_gpu = ops.Custom(
            iou_path,
            out_shape=lambda boxes, thresh: ((boxes[0],), thresh),
            out_dtype=(mstype.int64, mstype.int32),
            func_type="aot",
        )

    def construct(self, boxes, thresh):
        return self.nms_gpu(boxes, thresh)


# nmsnormalgpu
# in1:  boox [N, 7] float tensor,  in2: thresh [1], float tensor
# out1: keep [N]    long tensor,   out2: num_to_keep [1], int tensor
class NmsNormalGpu(Cell):
    def __init__(self):
        super(NmsNormalGpu, self).__init__()
        self.nms_normal_gpu = ops.Custom(
            "./iou_nms.so:NmsNormalGpu",
            out_shape=lambda boxes, thresh: ((boxes[0],), thresh),
            out_dtype=(mstype.int64, mstype.int32),
            func_type="aot",
        )

    def construct(self, boxes, thresh):
        return self.nms_normal_gpu(boxes, thresh)


if __name__ == "__main__":
    # 1、boxesiou
    print("[Python] ===========boxesiou start===========")
    a = np.loadtxt("boxes_iou_in_boxes_a_(52640, 7).txt", dtype=np.float32)
    b = np.loadtxt("boxes_iou_in_boxes_b_(2, 7).txt", dtype=np.float32)

    net = BoxesIouBevGpu()
    output = net(Tensor(a), Tensor(b))

    outdata = output.asnumpy()
    np.savetxt("boxes_iou_outz_" + str(outdata.shape) + "_my.txt", outdata, "%.6f")
    export(
        net, Tensor(a), Tensor(b), file_name="mi_boxes_iou_gpu", file_format="MINDIR"
    )

    #   graph = load("boxes_iou.mindir")
    #   net = nn.GraphCell(graph)
    #   output = net(Tensor(a), Tensor(b))

    #   outdata = output.asnumpy()
    #   np.savetxt('boxes_iou_outz_mindir' + str(outdata.shape) + '_my.txt', outdata, '%.6f')

    # 2、boxesoverlap
    print("[Python] ===========boxesoverlap start===========")
    a = np.loadtxt("boxes_overlap_in_boxes_a_(52640, 7).txt", dtype=np.float32)
    b = np.loadtxt("boxes_overlap_in_boxes_b_(6, 7).txt", dtype=np.float32)

    net = BoxesOverlapBevGpu()
    output = net(Tensor(a), Tensor(b))

    outdata = output.asnumpy()
    np.savetxt("boxes_overlap_outz_" + str(outdata.shape) + "_my.txt", outdata, "%.6f")
    export(
        net,
        Tensor(a),
        Tensor(b),
        file_name="mi_boxes_overlap_gpu",
        file_format="MINDIR",
    )

    # 3、nmsgpu
    print("[Python] ===========nmsgpu start===========")
    boxes = np.loadtxt("nms_gpu_in_boxes_(6, 7).txt", dtype=np.float32)
    thresh = np.array([0.01], dtype=np.float32)

    net = NumGpu()
    keep_array, num_array = net(Tensor(boxes), Tensor(thresh))
    num = num_array.asnumpy()[0]
    print("nms_out num:", num)
    outdata = keep_array.asnumpy()[:num]
    np.savetxt("nms_gpu_outz_" + str(outdata.shape) + "_my.txt", outdata, "%.6f")
    export(
        net, Tensor(boxes), Tensor(thresh), file_name="mi_num_gpu", file_format="MINDIR"
    )

    # 4、nmsnormalgpu
    print("[Python] ===========nmsnormalgpu start===========")
    boxes = np.loadtxt("nms_gpu_in_boxes_(6, 7).txt", dtype=np.float32)
    thresh = np.array([0.01], dtype=np.float32)

    net = NmsNormalGpu()
    keep_array, num_array = net(Tensor(boxes), Tensor(thresh))
    num = num_array.asnumpy()[0]
    print("nms_normal_out num:", num)
    outdata = keep_array.asnumpy()[:num]
    np.savetxt("nms_normal_gpu_outz_" + str(outdata.shape) + "_my.txt", outdata, "%.6f")
    export(
        net,
        Tensor(boxes),
        Tensor(thresh),
        file_name="mi_num_normal_gpu",
        file_format="MINDIR",
    )
