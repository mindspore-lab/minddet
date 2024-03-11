from typing import Optional, List, Tuple
from mindspore import Tensor, ops, nn
import mindspore as ms


def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = ops.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = ops.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


class GatherFeature(nn.Cell):
    """
    Gather feature at specified position

    Args:
        enable_cpu_gather (bool): Use cpu operator GatherD to gather feature or not, adaption for CPU. Default: False.

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self, enable_cpu_gather=False):
        super(GatherFeature, self).__init__()
        self.tile = ops.Tile()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.enable_cpu_gather = enable_cpu_gather
        self.start = Tensor(0, ms.int32)
        self.step = Tensor(1, ms.int32)
        if self.enable_cpu_gather:
            self.gather_nd = ops.GatherD()
            self.expand_dims = ops.ExpandDims()
        else:
            self.gather_nd = ops.GatherNd()

    def construct(self, feat, ind):
        """gather by specified index"""
        if self.enable_cpu_gather:
            _, _, c = self.shape(feat)
            # (b, N, c)
            index = self.expand_dims(ind, -1)
            index = self.tile(index, (1, 1, c))
            feat = self.gather_nd(feat, 1, index)
        else:
            # (b, N)->(b*N, 1)
            b, N = self.shape(ind)
            ind = self.reshape(ind, (-1, 1))
            ind_b = ops.range(self.start, Tensor(b, ms.int32), self.step)
            ind_b = self.reshape(ind_b, (-1, 1))
            ind_b = self.tile(ind_b, (1, N))
            ind_b = self.reshape(ind_b, (-1, 1))
            index = self.concat((ind_b, ind))
            # (b, N, 2)
            index = self.reshape(index, (b, N, -1))
            # (b, N, c)
            feat = self.gather_nd(feat, index)
        return feat


class TransposeGatherFeature(nn.Cell):
    """
    Transpose and gather feature at specified position

    Args: None

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self):
        super(TransposeGatherFeature, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 3, 1)
        self.gather_feat = GatherFeature()

    def construct(self, feat, ind):
        # (b, c, h, w)->(b, h*w, c)
        feat = self.transpose(feat, self.perm_list)
        b, _, _, c = self.shape(feat)
        feat = self.reshape(feat, (b, -1, c))
        # (b, N, c)
        feat = self.gather_feat(feat, ind)
        return feat


class DecodeFeatures(nn.Cell):
    def __init__(self):
        super(DecodeFeatures, self).__init__()
        self.trans_and_gather = TransposeGatherFeature()

    def predict_by_feat(self,
                        center_hm_preds,
                        wh_preds,
                        offset_preds,
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = True,
                        with_nms: bool = False):
        assert len(center_hm_preds) == len(wh_preds) == len(
            offset_preds) == 1
        result_list = []
        for img_id in range(len(batch_img_metas)):
            result_list.append(
                self._predict_by_feat_single(
                    center_hm_preds[0][img_id:img_id + 1, ...],  # (1, 80, 112, 168)
                    wh_preds[0][img_id:img_id + 1, ...],  # (1, 2, 112, 168)
                    offset_preds[0][img_id:img_id + 1, ...],
                    batch_img_metas[img_id],
                    rescale=rescale,  # True
                    with_nms=with_nms))  # False
        return result_list

    def _predict_by_feat_single(self,
                                center_heatmap_pred: Tensor,
                                wh_pred: Tensor,
                                offset_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True,
                                with_nms: bool = False):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        batch_det_bboxes, batch_labels = self._decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta['batch_input_shape'],  # (448, 672)
            k=100,  # {'topk': 100, 'local_maximum_kernel': 3, 'max_per_img': 100}
            kernel=3)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
                                                                 [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border

        if rescale and 'scale_factor' in img_meta:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = {'bboxes': det_bboxes[..., :4], 'scores': det_bboxes[..., 4], 'labels': det_labels}
        return results

    def _decode_heatmap(self,
                        center_heatmap_pred: Tensor,
                        wh_pred: Tensor,
                        offset_pred: Tensor,
                        img_shape: tuple,
                        k: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor]:
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (tuple): image shape in hw format.
            k (int): Get top k center keypoints from heatmap. Defaults to 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Defaults to 3.

        Returns:
            tuple[Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = self.trans_and_gather(wh_pred, batch_index)
        offset = self.trans_and_gather(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = ops.stack([tl_x, tl_y, br_x, br_y], axis=2)
        batch_bboxes = ops.cat((batch_bboxes, batch_scores[..., None]),
                               axis=-1)
        return batch_bboxes, batch_topk_labels
