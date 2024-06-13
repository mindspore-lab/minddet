import mindspore.ops as ops
from det3d_ms.ops.nms_with_mask3d import nms3d
from mindspore import context, nn
from mindspore.common import dtype as mstype
from mindspore.ops import CustomRegOp, DataType, custom_info_register

context.set_context(device_target="Ascend")


@custom_info_register(
    CustomRegOp()
    .input(0, "box_scores")
    .output(0, "selected_mask")
    .attr("iou_thr", "required", "float")
    .dtype_format(DataType.F32_Default, DataType.U8_Default)
    .target("Ascend")
    .get_op_info()
)
def nms_with_mask3d_core(
    box_scores, selected_mask, iou_thr, kernel_name="nms_with_mask3d"
):
    tik_instance = nms3d(box_scores, selected_mask, iou_thr, kernel_name)
    print(tik_instance)


class NMS(nn.Cell):
    def __init__(self):
        super(NMS, self).__init__()
        self.nms = ops.Custom(
            nms_with_mask3d_core,
            out_shape=lambda x, _: [x[1]],
            out_dtype=lambda x, _: mstype.uint8,
            func_type="tbe",
        )
        self.topK = ops.TopK()

    def construct(self, boxes, iou_thr):
        boxes = ops.Transpose()(boxes, (1, 0))
        nms_ret = self.nms(boxes, iou_thr)
        nms_ret = self.cast(nms_ret, mstype.float32)
        keep = self.topK(nms_ret, boxes.shape[1])[1]
        num_out = self.cast(nms_ret.sum(), mstype.int32)
        return keep, num_out
