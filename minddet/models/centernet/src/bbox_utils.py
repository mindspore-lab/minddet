from typing import Tuple, List
import numpy as np


def translate(boxes, distances):
    """Translate boxes in-place.

    Args:
        distances (Tuple[float, float]): translate distances. The first
            is horizontal distance and the second is vertical distance.
    """
    assert len(distances) == 2
    return boxes + np.array([distances[0], distances[1], distances[0], distances[1]])


def clip(boxes, img_shape):
    """Clip boxes according to the image shape in-place.

    Args:
        img_shape (Tuple[int, int]): A tuple of image height and width.
    """
    boxes[..., 0::2] = boxes[..., 0::2].clip(0, img_shape[1])
    boxes[..., 1::2] = boxes[..., 1::2].clip(0, img_shape[0])
    return boxes


def is_inside(boxes,
              img_shape,
              all_inside: bool = False,
              allowed_border: int = 0):
    """Find boxes inside the image.

    Args:
        img_shape (Tuple[int, int]): A tuple of image height and width.
        all_inside (bool): Whether the boxes are all inside the image or
            part inside the image. Defaults to False.
        allowed_border (int): Boxes that extend beyond the image shape
            boundary by more than ``allowed_border`` are considered
            "outside" Defaults to 0.
    Returns:
        BoolTensor: A BoolTensor indicating whether the box is inside
        the image. Assuming the original boxes have shape (m, n, 4),
        the output has shape (m, n).
    """
    img_h, img_w = img_shape
    if all_inside:
        return (boxes[:, 0] >= -allowed_border) & \
               (boxes[:, 1] >= -allowed_border) & \
               (boxes[:, 2] < img_w + allowed_border) & \
               (boxes[:, 3] < img_h + allowed_border)
    else:
        return (boxes[..., 0] < img_w + allowed_border) & \
               (boxes[..., 1] < img_h + allowed_border) & \
               (boxes[..., 2] > -allowed_border) & \
               (boxes[..., 3] > -allowed_border)


def rescale(boxes, scale_factor):
    assert len(scale_factor) == 2
    scale_factor = np.array([scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]])
    return boxes * scale_factor
