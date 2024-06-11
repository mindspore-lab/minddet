from mindspore import ops
from mindspore.common import dtype as mstype


# @numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    print(scores, scores.shape)
    scores = ops.Cast()(scores, mstype.float16)
    order = ops.Cast()(
        ops.Sort()(scores.view(-1))[0], mstype.int32
    )  # [::-1] scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = ops.Zeros()((ndets), mstype.int32)  # np.zeros((ndets), dtype=np.int32)
    print("suppressed:", suppressed.shape)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == ops.ZerosLike()(
            suppressed[i]
        ):  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep
