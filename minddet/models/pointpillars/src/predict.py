"""predict"""
from time import time

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import numpy as mnp
from mindspore import ops
from src.core import box_ops, nms
from src.data import kitti_common as kitti


def get_index_by_mask(mask):
    """get index by mask"""
    if isinstance(mask, np.ndarray):
        return np.where(mask)[0]
    return Tensor(np.where(mask.asnumpy()))[0]


def xor(a, b):
    """xor"""
    return Tensor(a.asnumpy() ^ b.asnumpy())


def _get_top_scores_labels(top_scores, top_labels, nms_score_threshold):
    """get top scores"""
    type = top_scores.dtype
    if nms_score_threshold > 0.0:
        if isinstance(top_scores, np.ndarray):  # for nms_score_thre type is np
            thresh = np.array([nms_score_threshold], dtype=type)
            top_scores_keep = top_scores >= thresh
        else:
            thresh = Tensor([nms_score_threshold], dtype=type)
            top_scores_keep = (top_scores >= thresh).astype(mstype.float16)
        if top_scores_keep.sum() > 0:
            if isinstance(top_scores, np.ndarray):
                top_scores = top_scores[top_scores_keep]
            else:
                top_scores_keep = get_index_by_mask(top_scores_keep)
                top_scores = top_scores[top_scores_keep]

    return top_scores, top_labels, top_scores_keep


def _get_selected_data(total_scores, box_preds, top_labels, dir_labels, cfg):
    """get selected data"""
    selected_boxes = None
    selected_labels = None
    selected_scores = None
    selected_dir_labels = None
    # get highest score per prediction, then apply nms
    # to remove overlapped box.
    top_scores, top_labels, top_scores_keep = _get_top_scores_labels(
        total_scores, top_labels, cfg["nms_score_threshold"]
    )
    box_preds = box_preds[top_scores_keep]
    if box_preds.shape[0] > 0:
        if cfg["nms_score_threshold"] > 0.0:
            if cfg["use_direction_classifier"]:
                dir_labels = dir_labels[top_scores_keep]
            top_labels = top_labels[top_scores_keep]

        boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
        # order by score
        scores = np.expand_dims(top_scores, 1)
        boxes_for_nms = boxes_for_nms.reshape(
            boxes_for_nms.shape[0], boxes_for_nms.shape[-1]
        )
        box_preds_corners = box_ops.center_to_corner_box2d(
            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4]
        )
        start_corner_to_standup = time()
        if isinstance(box_preds_corners, np.ndarray):
            boxes_for_nms = box_ops.corner_to_standup_nd_np_new(box_preds_corners)
            selected = nms.nms_np(
                boxes_for_nms,
                scores,
                pre_max_size=cfg["nms_pre_max_size"],
                post_max_size=cfg["nms_post_max_size"],
                iou_threshold=cfg["nms_iou_threshold"],
            )
        else:
            boxes_for_nms = box_ops.corner_to_standup_nd(box_preds_corners)
            # the nms in 3d detection just remove overlap boxes.
            selected = nms.nms_ops(
                boxes_for_nms,
                top_scores,
                pre_max_size=cfg["nms_pre_max_size"],
                post_max_size=cfg["nms_post_max_size"],
                iou_threshold=cfg["nms_iou_threshold"],
            )
    else:
        selected = None
    if selected is not None:
        selected_boxes = box_preds[selected]
        selected_labels = top_labels[selected]
        selected_scores = top_scores[selected]
        if cfg["use_direction_classifier"]:
            selected_dir_labels = dir_labels[selected]

    return selected_boxes, selected_labels, selected_scores, selected_dir_labels


def softmax(x, axis=1):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    prob_x = x_exp / x_sum
    return prob_x


def sigmoid(x):
    x = 1 + (1 / np.exp(x))
    prob_x = 1 / x
    return prob_x


def _get_total_scores(cls_preds, cfg):
    """get total scores"""
    cls_preds = ops.Tensor.from_numpy(cls_preds).float()
    if cfg["encode_background_as_zeros"]:
        if isinstance(cls_preds, np.ndarray):
            total_scores = sigmoid(cls_preds)
        else:
            total_scores = ops.Sigmoid()(cls_preds)
            total_scores = total_scores.asnumpy()
    else:
        # encode background as first element in one-hot vector
        if cfg["use_sigmoid_score"]:
            if isinstance(cls_preds, np.ndarray):
                total_scores = sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = ops.Sigmoid()(cls_preds)[..., 1:]
        else:
            if isinstance(cls_preds, np.ndarray):
                total_scores = softmax(cls_preds, -1)[..., 1:]
            else:
                total_scores = ops.Softmax(axis=-1)(cls_preds)[..., 1:]

    return total_scores


def get_template_prediction(num_samples):
    ret_dict = {
        "name": np.zeros(num_samples),
        "truncated": np.zeros(num_samples),
        "occluded": np.zeros(num_samples),
        "alpha": np.zeros(num_samples),
        "bbox": np.zeros([num_samples, 4]),
        "dimensions": np.zeros([num_samples, 3]),
        "location": np.zeros([num_samples, 3]),
        "rotation_y": np.zeros(num_samples),
        "score": np.zeros(num_samples),
        "boxes_lidar": np.zeros([num_samples, 7]),
    }
    return ret_dict


# generate anno
def generate_single_sample_dict(
    selected_data, rect, trv2c, p2, image_shape, img_idx, class_names
):
    (
        selected_boxes,
        selected_labels,
        selected_scores,
        selected_dir_labels,
    ) = selected_data
    pred_dict = {}
    if selected_boxes is not None:
        final_box_preds_camera = box_ops.box_lidar_to_camera(
            selected_boxes, rect, trv2c, False
        )
        box_2d_preds = box_ops.boxes3d_kitti_camera_to_imageboxes(
            final_box_preds_camera, p2
        )
        if image_shape is not None:
            box_2d_preds[:, 0] = np.clip(
                box_2d_preds[:, 0], a_min=0, a_max=image_shape[1] - 1
            )
            box_2d_preds[:, 1] = np.clip(
                box_2d_preds[:, 1], a_min=0, a_max=image_shape[0] - 1
            )
            box_2d_preds[:, 2] = np.clip(
                box_2d_preds[:, 2], a_min=0, a_max=image_shape[1] - 1
            )
            box_2d_preds[:, 3] = np.clip(
                box_2d_preds[:, 3], a_min=0, a_max=image_shape[0] - 1
            )

        pred_dict["name"] = np.array(class_names)[selected_labels]
        pred_dict["alpha"] = (
            -np.arctan2(-selected_boxes[:, 1], selected_boxes[:, 0])
            + final_box_preds_camera[:, 6]
        )
        pred_dict["bbox"] = box_2d_preds
        pred_dict["dimensions"] = final_box_preds_camera[:, 3:6]
        pred_dict["location"] = final_box_preds_camera[:, 0:3]
        pred_dict["rotation_y"] = final_box_preds_camera[:, 6]
        pred_dict["score"] = selected_scores
        pred_dict["boxes_lidar"] = selected_boxes
        pred_dict["image_idx"] = img_idx
    else:
        pred_dict = kitti.empty_result_anno()
    return pred_dict


def generate_single_sample_dict_old(
    selected_data,
    rect,
    trv2c,
    p2,
    img_idx,
    use_direction_classifier,
    image_shape,
    class_names,
    limit_range,
):
    (
        selected_boxes,
        selected_labels,
        selected_scores,
        selected_dir_labels,
    ) = selected_data
    if selected_boxes is not None:
        if use_direction_classifier:
            if isinstance(selected_boxes, np.ndarray):
                opp_labels = (selected_boxes[..., -1] > 0) ^ selected_dir_labels.astype(
                    np.bool_
                )
            else:
                opp_labels = xor((selected_boxes[..., -1] > 0), selected_dir_labels)
            if isinstance(opp_labels, np.ndarray):
                selected_boxes[..., -1] += np.where(opp_labels, np.pi, 0.0)
            else:
                selected_boxes[..., -1] += mnp.where(
                    opp_labels,
                    Tensor(mnp.pi, dtype=selected_boxes.dtype),
                    Tensor(0.0, dtype=selected_boxes.dtype),
                )
        final_box_preds_camera = box_ops.box_lidar_to_camera(
            selected_boxes, rect, trv2c
        )
        box_2d_preds = box_ops.boxes3d_kitti_camera_to_imageboxes(
            final_box_preds_camera, p2
        )

        pred_dict = {
            "bbox": box_2d_preds,
            "box3d_camera": final_box_preds_camera,
            "box3d_lidar": selected_boxes,
            "scores": selected_scores,
            "label_preds": selected_labels,
            "image_idx": img_idx,
        }
        # pred_dict['name'] = np.array(class_names)[selected_labels]
        # pred_dict['alpha'] = -np.arctan2(-selected_boxes[:, 1], selected_boxes[:, 0]) + final_box_preds_camera[:, 6]
        # pred_dict['bbox'] = box_2d_preds
        # pred_dict['dimensions'] = final_box_preds_camera[:, 3:6]
        # pred_dict['location'] = final_box_preds_camera[:, 0:3]
        # pred_dict['rotation_y'] = final_box_preds_camera[:, 6]
        # pred_dict['score'] = selected_scores
        # pred_dict['boxes_lidar'] = selected_boxes
        # pred_dict['image_idx'] = img_idx
    else:
        pred_dict = {
            "bbox": None,
            "box3d_camera": None,
            "box3d_lidar": None,
            "scores": None,
            "label_preds": None,
            "image_idx": img_idx,
        }
    return pred_dict


def predict(
    preds_dict,
    example,
    batch_image_shape,
    cfg,
    class_names,
    use_self_train=True,
    center_limit_range=None,
):
    use_direction_classifier = cfg["use_direction_classifier"]
    batch_rect = example["rect"]
    batch_trv2c = example["Trv2c"]
    batch_p2 = example["P2"]
    batch_imgidx = example["image_idx"]
    if not isinstance(batch_image_shape, np.ndarray):
        batch_image_shape = batch_image_shape.asnumpy()

    predictions_dicts = []

    for pred, rect, trv2c, p2, img_idx, image_shape in zip(
        preds_dict, batch_rect, batch_trv2c, batch_p2, batch_imgidx, batch_image_shape
    ):
        box_preds, cls_preds, scores, indices, dir_preds = pred
        top_indices = indices.asnumpy()
        top_labels = cls_preds.asnumpy()
        top_labels = top_labels[top_indices]
        top_scores = scores.asnumpy()
        dir_preds = dir_preds.asnumpy()
        dir_preds = dir_preds[top_indices]
        top_indices = np.expand_dims(top_indices, 1)
        box_preds = box_preds.asnumpy()
        box_preds = box_preds[top_indices]
        box_preds = box_preds.reshape(box_preds.shape[0], box_preds.shape[-1])

        selected_data = _get_selected_data(
            top_scores, box_preds, top_labels, dir_preds, cfg
        )
        if use_self_train:
            predictions_dict = generate_single_sample_dict_old(
                selected_data,
                rect,
                trv2c,
                p2,
                img_idx,
                use_direction_classifier,
                image_shape,
                class_names,
                center_limit_range,
            )
        else:
            predictions_dict = generate_single_sample_dict(
                selected_data, rect, trv2c, p2, image_shape, img_idx, class_names
            )

        predictions_dicts.append(predictions_dict)
    return predictions_dicts


def predict_kitti_to_anno(
    predictions_dicts,
    batch_image_shape,
    class_names,
    center_limit_range=None,
    lidar_input=False,
):
    """predict kitti to anno"""
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        for k, v in preds_dict.items():
            if v is not None:
                # if isinstance(v, np.int64):
                if isinstance(batch_image_shape, np.ndarray):
                    preds_dict[k] = v
                else:
                    preds_dict[k] = v.asnumpy()
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"]
            box_preds = preds_dict["box3d_camera"]
            scores = preds_dict["scores"]
            box_preds_lidar = preds_dict["box3d_lidar"]
            # write pred to file
            label_preds = preds_dict["label_preds"]
            anno = kitti.get_start_result_anno()
            num_example = 0

            for box, box_lidar, bbox, score, label in zip(
                box_preds, box_preds_lidar, box_2d_preds, scores, label_preds
            ):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if np.any(box_lidar[:3] < limit_range[:3]) or np.any(
                        box_lidar[:3] > limit_range[3:]
                    ):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array([img_idx] * num_example, dtype=np.int64)
    return annos
