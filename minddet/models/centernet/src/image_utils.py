from math import sqrt

import numpy as np

# def gaussian2D(shape, sigma=1):
#     """2D gaussian function"""
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y, x = np.ogrid[-m:m+1, -n:n+1]
#
#     h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     return h


def gaussian2D(radius, sigma=1, dtype=np.float32):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = np.arange(-radius, radius + 1, dtype=dtype).reshape(1, -1)
    y = np.arange(-radius, radius + 1, dtype=dtype).reshape(-1, 1)

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap):
    """
    Generate 2D gaussian radius.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(radius, sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian_kernel[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    out_heatmap = heatmap
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return out_heatmap
