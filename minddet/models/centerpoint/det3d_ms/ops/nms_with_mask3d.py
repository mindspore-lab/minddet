#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
nms3d
"""
import numpy as np
import te.platform as tbe_platform
from det3d_ms.ops.iou_utils import BevIouUB
from det3d_ms.ops.iou_utils import Constant as IouConstant
from tbe.common.platform import set_current_compile_soc_info
from te import tik
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """

    # shape's dim of input must be 2
    INPUT_DIM = 2
    # scaling factor
    DOWN_FACTOR = 0.054395
    # vector unit can compute 256 bytes in one cycle
    BYTES_ONE_CYCLE_VECTOR = 256
    # process 128 proposals at a time for fp16
    BURST_PROPOSAL_NUM = 128
    # RPN compute 16 proposals per iteration
    RPN_PROPOSAL_NUM = 16
    # the coordinate column contains x1,y1,x2,y2
    COORD_COLUMN_NUM = 4
    # valid proposal column contains x,y,theta,dx,dy
    VALID_COLUMN_NUM = 5
    # each region proposal contains eight elements
    ELEMENT_NUM = 8
    # data align size, also size of one block
    CONFIG_DATA_ALIGN = 32
    REPEAT_TIMES_MAX = 255
    # next_nonzero_idx shape0 is 16 for 32B aligned, 16 is enough
    SHAPE_NEXT_NONZERO = 16
    # mask used for vcmax in update_next_nonzero, 256//2=128, fixed fp16 here but enough for input_dtype
    MASK_VCMAX_FP16 = 128
    REPEAT_ELEM_FP32 = 64
    FP32_SIZE = tbe_platform.get_bit_len("fp32") // 8


def _ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def _ceiling(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value and can divide factor
    """
    result = (value + (factor - 1)) // factor * factor
    return result


def _tik_func_nms_single_core_multithread(
    input_shape, thresh, total_output_proposal_num, kernel_name_var
):
    """
    Compute output boxes after non-maximum suppression.

    Parameters
    ----------
    input_shape: dict
        shape of input boxes, including proposal boxes and corresponding confidence scores

    thresh: float
        iou threshold

    total_output_proposal_num: int
        the number of output proposal boxes

    kernel_name_var: str
        cce kernel name

    Returns
    -------
    tik_instance: TIK API
    """
    tik_instance = tik.Tik()
    _, total_input_proposal_num = input_shape

    proposals_fp32 = tik_instance.Tensor(
        "float32",
        (Constant.ELEMENT_NUM, total_input_proposal_num),
        name="in_proposals_fp32",
        scope=tik.scope_gm,
    )

    # output shape is [N]
    out_mask = tik_instance.Tensor(
        "uint8", (total_output_proposal_num,), name="out_mask", scope=tik.scope_gm
    )
    # address is 32B aligned
    output_mask_ub = tik_instance.Tensor(
        "uint8",
        (Constant.BURST_PROPOSAL_NUM,),
        name="output_mask_ub",
        scope=tik.scope_ubuf,
    )

    temp_8_N_ub = tik_instance.Tensor(
        "float32",
        (Constant.ELEMENT_NUM, Constant.BURST_PROPOSAL_NUM),
        name="temp_8_N_ub",
        scope=tik.scope_ubuf,
    )

    # init middle selected 8,N
    selected_3diou_ub = tik_instance.Tensor(
        "float32",
        (
            Constant.ELEMENT_NUM,
            _ceiling(total_output_proposal_num, Constant.RPN_PROPOSAL_NUM),
        ),
        name="selected_3diou_ub",
        scope=tik.scope_ubuf,
    )

    tik_instance.h_duplicate(selected_3diou_ub, 0.0)

    # init middle sup_vec
    sup_vec_ub = tik_instance.Tensor(
        "uint16",
        (_ceiling(total_output_proposal_num, Constant.RPN_PROPOSAL_NUM),),
        name="sup_vec_ub",
        scope=tik.scope_ubuf,
    )
    tik_instance.vector_dup(16, sup_vec_ub[0], 1, 1, 1, 8)

    temp_sup_matrix_ub = tik_instance.Tensor(
        "uint16",
        (_ceiling(total_output_proposal_num, Constant.RPN_PROPOSAL_NUM),),
        name="temp_sup_matrix_ub",
        scope=tik.scope_ubuf,
    )
    temp_sup_vec_ub = tik_instance.Tensor(
        "uint16",
        (Constant.BURST_PROPOSAL_NUM,),
        name="temp_sup_vec_ub",
        scope=tik.scope_ubuf,
    )

    # init ori coord
    mask = tik_instance.Scalar(dtype="uint8")

    # variables
    selected_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    selected_proposals_cnt.set_as(0)
    handling_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    handling_proposals_cnt.set_as(0)
    left_proposal_cnt = tik_instance.Scalar(dtype="uint16")
    left_proposal_cnt.set_as(total_input_proposal_num)  # [N,8] 中的　Ｎ
    scalar_zero = tik_instance.Scalar(dtype="uint16")
    scalar_zero.set_as(0)
    sup_vec_ub[0].set_as(scalar_zero)
    op_obj = BevIouUB(tik_instance, thresh)

    # handle 128 proposals every time  (x,y,theta,dx,dy,dz,none,none)
    with tik_instance.for_range(
        0,
        _ceil_div(total_input_proposal_num, Constant.BURST_PROPOSAL_NUM),
        thread_num=1,
    ) as burst_index:
        # update counter
        with tik_instance.if_scope(left_proposal_cnt < Constant.BURST_PROPOSAL_NUM):
            handling_proposals_cnt.set_as(left_proposal_cnt)
        with tik_instance.else_scope():
            handling_proposals_cnt.set_as(Constant.BURST_PROPOSAL_NUM)
        # 128 proposals
        burst = _ceil_div(
            handling_proposals_cnt * Constant.FP32_SIZE, Constant.CONFIG_DATA_ALIGN
        )
        tik_instance.vector_dup(
            Constant.REPEAT_ELEM_FP32,
            temp_8_N_ub,
            0,
            Constant.BURST_PROPOSAL_NUM
            * Constant.ELEMENT_NUM
            // Constant.REPEAT_ELEM_FP32,
            1,
            8,
        )
        with tik_instance.for_range(0, Constant.ELEMENT_NUM - 1) as move_idx:
            with tik_instance.if_scope(
                tik.any(
                    move_idx == IouConstant.X_IDX,
                    move_idx == IouConstant.Y_IDX,
                    move_idx == IouConstant.W_IDX,
                    move_idx == IouConstant.H_IDX,
                    move_idx == IouConstant.T_IDX,
                )
            ):
                tik_instance.data_move(
                    temp_8_N_ub[move_idx, 0],
                    proposals_fp32[move_idx, burst_index * Constant.BURST_PROPOSAL_NUM],
                    0,
                    1,
                    burst,
                    0,
                    0,
                )

        # start to update iou and or area from the first 16 proposal and get suppression vector 16 by 16 proposal
        length = tik_instance.Scalar(dtype="uint16")
        length.set_as(_ceiling(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM))

        # clear temp_sup_vec_ub
        tik_instance.vector_dup(
            128,
            temp_sup_vec_ub[0],
            1,
            temp_sup_vec_ub.shape[0] // Constant.BURST_PROPOSAL_NUM,
            1,
            8,
        )

        with tik_instance.for_range(
            0, _ceil_div(handling_proposals_cnt, Constant.RPN_PROPOSAL_NUM)
        ) as i:
            length.set_as(length + Constant.RPN_PROPOSAL_NUM)
            with tik_instance.for_range(
                0, _ceil_div(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM)
            ) as iou_index:
                op_obj.compute(
                    temp_8_N_ub[
                        :,
                        i
                        * Constant.RPN_PROPOSAL_NUM : (i + 1)
                        * Constant.RPN_PROPOSAL_NUM,
                    ],
                    selected_3diou_ub[
                        :,
                        iou_index
                        * Constant.RPN_PROPOSAL_NUM : (iou_index + 1)
                        * Constant.RPN_PROPOSAL_NUM,
                    ],
                    temp_sup_matrix_ub[
                        iou_index
                        * Constant.RPN_PROPOSAL_NUM : (iou_index + 1)
                        * Constant.RPN_PROPOSAL_NUM
                    ],
                )
            with tik_instance.for_range(0, i + 1) as iou_index:
                offset = _ceiling(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM)
                op_obj.compute(
                    temp_8_N_ub[
                        :,
                        i
                        * Constant.RPN_PROPOSAL_NUM : (i + 1)
                        * Constant.RPN_PROPOSAL_NUM,
                    ],
                    temp_8_N_ub[
                        :,
                        iou_index
                        * Constant.RPN_PROPOSAL_NUM : (iou_index + 1)
                        * Constant.RPN_PROPOSAL_NUM,
                    ],
                    temp_sup_matrix_ub[
                        (offset + iou_index * Constant.RPN_PROPOSAL_NUM) : (
                            (iou_index + 1) * Constant.RPN_PROPOSAL_NUM + offset
                        )
                    ],
                )

            rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
            # non-diagonal
            rpn_cor_ir = tik_instance.rpn_cor(
                temp_sup_matrix_ub[0],
                sup_vec_ub[0],
                1,
                1,
                _ceil_div(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM),
            )
            with tik_instance.if_scope(i > 0):
                rpn_cor_ir = tik_instance.rpn_cor(
                    temp_sup_matrix_ub[
                        _ceiling(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM)
                    ],
                    temp_sup_vec_ub,
                    1,
                    1,
                    i,
                )
            # diagonal
            tik_instance.rpn_cor_diag(
                temp_sup_vec_ub[i * Constant.RPN_PROPOSAL_NUM],
                temp_sup_matrix_ub[length - Constant.RPN_PROPOSAL_NUM],
                rpn_cor_ir,
            )

        # tik_instance.h_duplicate(output_mask_ub, 0)
        with tik_instance.for_range(0, handling_proposals_cnt) as i:
            with tik_instance.if_scope(
                tik.all(
                    temp_sup_vec_ub[i] == 0,
                    temp_8_N_ub[IouConstant.W_IDX, i] > 0,
                    temp_8_N_ub[IouConstant.H_IDX, i] > 0,
                )
            ):
                with tik_instance.for_range(0, Constant.ELEMENT_NUM - 1) as j:
                    selected_3diou_ub[j, selected_proposals_cnt].set_as(
                        temp_8_N_ub[j, i]
                    )
                sup_vec_ub[selected_proposals_cnt].set_as(scalar_zero)
                mask.set_as(1)
                output_mask_ub[i].set_as(mask)
                # update counter
                selected_proposals_cnt.set_as(selected_proposals_cnt + 1)
            with tik_instance.else_scope():
                mask.set_as(0)
                output_mask_ub[i].set_as(mask)

        left_proposal_cnt.set_as(left_proposal_cnt - handling_proposals_cnt)
        # mov target proposals to out - mte3

        tik_instance.data_move(
            out_mask[burst_index * Constant.BURST_PROPOSAL_NUM],
            output_mask_ub,
            0,
            1,
            _ceil_div(handling_proposals_cnt, Constant.CONFIG_DATA_ALIGN),
            0,
            0,
            0,
        )

    tik_instance.BuildCCE(
        kernel_name=kernel_name_var,
        inputs=[proposals_fp32],
        outputs=[out_mask],
        output_files_path=None,
        enable_l2=False,
    )
    return tik_instance


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
# @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
#                             para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def nms3d(box_scores, selected_mask, iou_thr, kernel_name="nms3d"):
    """
    algorithm: nms3d

    find the best target bounding box and eliminate redundant bounding boxes

    Parameters
    ----------
    box_scores: dict
        2-D shape and dtype of input tensor, only support [N, 8]
        including proposal boxes and corresponding confidence scores

    selected_boxes: dict
        2-D shape and dtype of output boxes tensor, only support [N,5]
        including proposal boxes and corresponding confidence scores

    selected_idx: dict
        the index of input proposal boxes

    selected_mask: dict
        the symbol judging whether the output proposal boxes is valid

    iou_thr: float
        iou threshold

    kernel_name: str
        cce kernel name, default value is "nms3d"

    Returns
    -------
    None
    """
    # check shape
    input_shape = box_scores.get("shape")
    # print("input_shape:")
    # print(input_shape)
    para_check.check_shape(
        input_shape,
        min_dim=1,
        min_rank=Constant.INPUT_DIM,
        max_rank=Constant.INPUT_DIM,
        param_name="box_scores",
    )

    input_dtype = box_scores.get("dtype").lower()

    # check dtype
    check_list = "float32"
    para_check.check_dtype(input_dtype, check_list, param_name="box_scores")

    # Considering the memory space of Unified_Buffer
    fp16_size = tbe_platform.get_bit_len("float16") // 8
    int32_size = tbe_platform.get_bit_len("int32") // 8
    uint8_size = tbe_platform.get_bit_len("uint8") // 8
    uint16_size = tbe_platform.get_bit_len("uint16") // 8
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # output shape is [N,5], including x1,y1,x2,y2,scores
    burst_size = (
        Constant.BURST_PROPOSAL_NUM * int32_size
        + Constant.BURST_PROPOSAL_NUM * uint8_size
        + Constant.BURST_PROPOSAL_NUM * Constant.VALID_COLUMN_NUM * fp16_size
    )
    # compute shape is [N,8]
    selected_size = (
        _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM)
        * Constant.ELEMENT_NUM
        * fp16_size
        + _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM) * fp16_size
        + _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM) * uint16_size
    )
    # intermediate calculation results
    temp_sup_matrix_size = (
        _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM) * uint16_size
    )
    temp_sup_vec_size = Constant.BURST_PROPOSAL_NUM * uint16_size
    temp_area_size = Constant.BURST_PROPOSAL_NUM * fp16_size
    temp_reduced_proposals_size = (
        Constant.BURST_PROPOSAL_NUM * Constant.ELEMENT_NUM * fp16_size
    )
    temp_size = (
        temp_sup_matrix_size
        + temp_sup_vec_size
        + temp_area_size
        + temp_reduced_proposals_size
    )
    # input shape is [N,8]
    fresh_size = Constant.BURST_PROPOSAL_NUM * Constant.ELEMENT_NUM * fp16_size

    coord_size = Constant.BURST_PROPOSAL_NUM * Constant.COORD_COLUMN_NUM * fp16_size
    used_size = burst_size + selected_size + temp_size + fresh_size + coord_size

    if used_size > ub_size_bytes:
        error_manager_vector.raise_err_check_params_rules(
            kernel_name,
            "the number of input boxes out of range(%d B)" % ub_size_bytes,
            "used size",
            used_size,
        )

    if input_shape[0] != Constant.ELEMENT_NUM:
        error_manager_vector.raise_err_check_params_rules(
            kernel_name,
            "the 2nd-dim of input boxes must be equal to 8",
            "box_scores.shape",
            input_shape,
        )

    _, output_size = input_shape
    return _tik_func_nms_single_core_multithread(
        input_shape, iou_thr, output_size, kernel_name
    )


def test_sample(mode):
    set_current_compile_soc_info("Ascend910")

    N = 32
    # proposals = np.full((N, 8), 1).astype(np.float32)
    proposals = np.fromfile(
        "/data/hxy/next_ads/code/training_ops/data/no-minus_8x32_for_aicore.bin",
        dtype=np.float32,
    ).reshape(8, N)
    # proposals = proposals.transpose(1, 0)
    # proposals[:, 1] = np.full(N, 2).astype(np.float32)
    # proposals[:, 2] = np.full(N, 3).astype(np.float32)
    # proposals[:, 3] = np.full(N, 4).astype(np.float32)
    print("proposals:\n{}".format(proposals))

    tik_instance = nms3d(
        {"shape": (8, N), "dtype": "float32"},
        {"shape": (N), "dtype": "uint8"},
        iou_thr=0.01,
        kernel_name="nms3d",
    )
    feed_dict = {"in_proposals_fp32": proposals}  # 对齐算子输入dict

    if mode == "pv":
        (out_mask,) = tik_instance.tikdb.start_debug(
            feed_dict=feed_dict, interactive=False
        )
        print("out_mask:\n{}".format(out_mask))

    elif mode == "ca":
        (out_mask,) = tik_instance.StartProfiling(
            feed_dict=feed_dict, simulatorlog_path="./ca_log"
        )
        print("out_mask:\n{}".format(out_mask))


if __name__ == "__main__":
    test_sample("pv")  # 可以修改此处为"ca"测试ca模型
