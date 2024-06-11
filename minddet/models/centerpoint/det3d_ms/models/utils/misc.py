import mindspore.numpy as mnp
from mindspore import ops
from mindspore.common import dtype as mstype

# from lib.models.backbone.utils import Registry
#
# BACKBONES = Registry()
# RPN_HEADS = Registry()
# ROI_BOX_FEATURE_EXTRACTORS = Registry()
# ROI_BOX_PREDICTOR = Registry()
# ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
# ROI_KEYPOINT_PREDICTOR = Registry()
# ROI_MASK_FEATURE_EXTRACTORS = Registry()
# ROI_MASK_PREDICTOR = Registry()


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = ops.ExpandDims()(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = mnp.arange(max_num).reshape(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = ops.Cast()(actual_num, mstype.int32) > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator
