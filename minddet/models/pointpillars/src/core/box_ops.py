"""box ops mindspore"""
import numpy as np
from mindspore import Tensor
from mindspore import numpy as mnp
from mindspore import ops
from src.core.einsum import einsum


def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for PointPillars in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    # need to convert boxes to z-center format
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = np.split(boxes, 7, axis=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = np.sqrt(la**2 + wa**2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal

    zt = (zg - za) / ha  # 1.6
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, zt, wt, lt, ht, rtx, rty], axis=-1)
    rt = rg - ra
    return np.concatenate([xt, yt, zt, wt, lt, ht, rt], axis=-1)


def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False):
    """box decode for VoxelNet in lidar
    Args:
        box_encodings ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    # need to convert box_encodings to z-bottom format
    xa, ya, za, wa, la, ha, ra = ops.Split(axis=-1, output_num=7)(anchors)
    rt = ops.Tensor(1)
    rtx = ops.Tensor(1)
    rty = ops.Tensor(1)
    if not encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rt = ops.Split(axis=-1, output_num=7)(box_encodings)
    else:
        xt, yt, zt, wt, lt, ht, rtx, rty = ops.Split(axis=-1, output_num=8)(
            box_encodings
        )
    za = za + ha / 2
    diagonal = ops.Sqrt()(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    lg = ops.Exp()(lt) * la
    wg = ops.Exp()(wt) * wa
    hg = ops.Exp()(ht) * ha

    if not encode_angle_to_vector:
        rg = rt + ra
    else:
        rax = ops.Cos()(ra)
        ray = ops.Sin()(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = ops.Atan2(rgy, rgx)
    zg = zg - hg / 2
    result = ops.Concat(axis=-1)([xg, yg, zg, wg, lg, hg, rg])
    return result


def bev_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for PointPillars
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    xa, ya, wa, la, ra = ops.Split(axis=-1)(anchors)
    xg, yg, wg, lg, rg = ops.Split(axis=-1)(boxes)
    diagonal = ops.Sqrt()(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
    else:
        lt = ops.Log()(lg / la)
        wt = ops.Log()(wg / wa)
    if encode_angle_to_vector:
        rgx = ops.Cos()(rg)
        rgy = ops.Sin()(rg)
        rax = ops.Cos()(ra)
        ray = ops.Sin()(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return ops.Concat(axis=-1)([xt, yt, wt, lt, rtx, rty])
    rt = rg - ra
    return ops.Concat(axis=-1)([xt, yt, wt, lt, rt])


def bev_box_decode_np(
    box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False
):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    # need to convert box_encodings to z-bottom format
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = np.split(box_encodings, 6, axis=-1)
    else:
        xt, yt, wt, lt, rt = np.split(box_encodings, 5, axis=-1)
    diagonal = np.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra
    return np.concatenate([xg, yg, wg, lg, rg], axis=-1)


def bev_box_decode(
    box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False
):
    """box decode for VoxelNet in lidar
    Args:
        box_encodings ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance, decrease other performance.
        smooth_dim: bool
    """
    xa, ya, wa, la, ra = ops.Split(axis=-1)(anchors)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = ops.Split(axis=-1)(box_encodings)

    else:
        xt, yt, wt, lt, rt = ops.Split(axis=-1)(box_encodings)

    diagonal = ops.Sqrt()(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = ops.Exp()(lt) * la
        wg = ops.Exp()(wt) * wa
    if encode_angle_to_vector:
        rax = ops.Cos()(ra)
        ray = ops.Sin()(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = ops.Atan2()(rgy, rgx)
    else:
        rg = rt + ra
    return ops.Concat(axis=-1)([xg, yg, wg, lg, rg])


def corners_nd_np(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(1, 2**ndim, ndim)
    return corners


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = ops.Stack(axis=1)(
        mnp.unravel_index(mnp.arange(2**ndim), (2,) * ndim)
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - Tensor(origin, dtype=dims.dtype)
    corners_norm = Tensor(corners_norm).astype(dims.dtype)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    return corners


def corner_to_standup_nd_np(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def corner_to_standup_nd_np_new(boxes_corner):
    """
    Convert bounding boxes from corner format to standup format.

    Args:
        boxes_corner (numpy.ndarray): An array of bounding boxes in corner format,
            shape is [N, 4, D], where N is the number of boxes, D is the number of
            dimensions (e.g. 2 for 2D boxes).

    Returns:
        numpy.ndarray: The converted bounding boxes, shape is [N, 2 * D].
    """
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)


def corner_to_standup_nd_new(boxes_corner):
    """corner to standup nd"""
    standup_boxes = []
    standup_boxes.append(ops.min(boxes_corner, axis=1))
    standup_boxes.append(ops.max(boxes_corner, axis=1))
    return ops.Stack(axis=1)(standup_boxes)


def corner_to_standup_nd(boxes_corner):
    """corner to standup nd"""
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(boxes_corner[:, :, i].min(axis=1))
    for i in range(ndim):
        standup_boxes.append(boxes_corner[:, :, i].max(axis=1))
    return ops.Stack(axis=1)(standup_boxes)


def rotation_3d_in_axis_np(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis in (2, -1):
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_3d_in_axis(points, angles, axis=0):
    """rotation 3d in axis"""
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = ops.Sin()(angles)
    rot_cos = ops.Cos()(angles)
    ones = ops.OnesLike()(rot_cos)
    zeros = ops.ZerosLike()(rot_cos)
    if axis == 1:
        rot_mat_t = ops.Stack()(
            [
                ops.Stack()([rot_cos, zeros, -rot_sin]),
                ops.Stack()([zeros, ones, zeros]),
                ops.Stack()([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis in (2, -1):
        rot_mat_t = ops.Stack()(
            [
                ops.Stack()([rot_cos, -rot_sin, zeros]),
                ops.Stack()([rot_sin, rot_cos, zeros]),
                ops.Stack()([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_t = ops.Stack()(
            [
                ops.Stack()([zeros, rot_cos, -rot_sin]),
                ops.Stack()([zeros, rot_sin, rot_cos]),
                ops.Stack()([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError("axis should in range")
    return einsum("aij,jka->aik", points, rot_mat_t)


def rotation_2d_np(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = ops.Sin()(angles)
    rot_cos = ops.Cos()(angles)
    rot_mat_t = ops.Stack()(
        [ops.Stack()([rot_cos, -rot_sin]), ops.Stack()([rot_sin, rot_cos])]
    )
    return einsum("aij,jka->aik", points, rot_mat_t)


def center_to_corner_box3d(centers, dims, angles, origin=(0.5, 1.0, 0.5), axis=1):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    if isinstance(centers, np.ndarray):
        corners = corners_nd_np(dims, origin=origin)
        corners = rotation_3d_in_axis_np(corners, angles, axis=axis)
        corners += centers.reshape(-1, 1, 3)
    else:
        corners = corners_nd(dims, origin=origin)
        # corners: [N, 8, 3]
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
        corners += centers.view(-1, 1, 3)
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    if isinstance(centers, np.ndarray):
        corners = corners_nd_np(dims, origin=origin)
    else:
        corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        if isinstance(centers, np.ndarray):
            corners = rotation_2d_np(corners, angles)
        else:
            corners = rotation_2d(corners, angles)
    if isinstance(centers, np.ndarray):
        corners += centers.reshape(-1, 1, 2)
    else:
        corners += centers.view(-1, 1, 2)
    return corners


def project_to_image_np(points_3d, proj_mat):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def project_to_image(points_3d, proj_mat):
    """project to image"""
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = ops.Concat(axis=-1)(
        [points_3d, ops.Zeros()(tuple(points_shape), points_3d.dtype)]
    )
    point_2d = ops.MatMul()(
        ops.Reshape()(points_4, (-1, proj_mat.T.shape[-1])), proj_mat.T
    )
    shape = (*points_4.shape[:2], proj_mat.T.shape[-1])
    point_2d = ops.Reshape()(point_2d, shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def lidar_to_camera_np(points, r_rect, velo2cam):
    """lidar to camera"""
    # num_points = points.shape[0]
    # points = ops.Concat(axis=-1)([points, ops.Ones()((num_points, 1), points.dtype)])
    # camera_points = ops.MatMul()(points, ops.MatMul()(r_rect, velo2cam).T)
    points_shape = list(points.shape[:-1])
    points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    """lidar to camera"""
    num_points = points.shape[0]
    points = ops.Concat(axis=-1)([points, ops.Ones()((num_points, 1), points.dtype)])
    camera_points = ops.MatMul()(points, ops.MatMul()(r_rect, velo2cam).T)
    return camera_points[..., :3]


def box_lidar_to_camera(data, r_rect, velo2cam, use_self_train=True):
    """box lidar to camera"""
    # if not use_self_train:
    #     boxes3d_lidar_copy = copy.deepcopy(data)
    #     xyz_lidar = boxes3d_lidar_copy[..., 0:3]
    #     l, w, h = boxes3d_lidar_copy[..., 3:4], boxes3d_lidar_copy[..., 4:5], boxes3d_lidar_copy[..., 5:6]
    #     r = boxes3d_lidar_copy[..., 6:7]
    #     xyz_lidar[:, 2] -= h.reshape(-1) / 2
    #     if isinstance(boxes3d_lidar_copy, np.ndarray):
    #         xyz = lidar_to_camera_np(xyz_lidar, r_rect, velo2cam)
    #     else:
    #         xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    #     r = -r - np.pi / 2
    #     if isinstance(xyz, np.ndarray):
    #         return np.concatenate([xyz, l, h, w, r], axis=1)
    #     return ops.Concat(axis=-1)([xyz, l, h, w, r])

    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    if isinstance(data, np.ndarray):
        xyz = lidar_to_camera_np(xyz_lidar, r_rect, velo2cam)
    else:
        xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    if isinstance(xyz, np.ndarray):
        return np.concatenate([xyz, l, h, w, r], axis=1)
    return ops.Concat(axis=-1)([xyz, l, h, w, r])


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array(
        [l / 2.0, l / 2.0, -l / 2.0, -l / 2.0, l / 2.0, l / 2.0, -l / 2.0, -l / 2],
        dtype=np.float32,
    ).T
    z_corners = np.array(
        [w / 2.0, -w / 2.0, -w / 2.0, w / 2.0, w / 2.0, -w / 2.0, -w / 2.0, w / 2.0],
        dtype=np.float32,
    ).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array(
            [
                h / 2.0,
                h / 2.0,
                h / 2.0,
                h / 2.0,
                -h / 2.0,
                -h / 2.0,
                -h / 2.0,
                -h / 2.0,
            ],
            dtype=np.float32,
        ).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(
        ry.size, dtype=np.float32
    )
    rot_list = np.array(
        [
            [np.cos(ry), zeros, -np.sin(ry)],
            [zeros, ones, zeros],
            [np.sin(ry), zeros, np.cos(ry)],
        ]
    )  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate(
        (
            x_corners.reshape(-1, 8, 1),
            y_corners.reshape(-1, 8, 1),
            z_corners.reshape(-1, 8, 1),
        ),
        axis=2,
    )  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = (
        rotated_corners[:, :, 0],
        rotated_corners[:, :, 1],
        rotated_corners[:, :, 2],
    )

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2
    )

    return corners.astype(np.float32)


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom


def rect_to_img(pts_rect, p2):
    """
    :param pts_rect: (N, 3)
    :return pts_img: (N, 2)
    """
    pts_rect_hom = cart_to_hom(pts_rect)
    pts_2d_hom = np.dot(pts_rect_hom, p2.T)
    pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
    pts_rect_depth = pts_2d_hom[:, 2] - p2.T[3, 2]  # depth in rect camera coord
    return pts_img, pts_rect_depth


def boxes3d_kitti_camera_to_imageboxes(boxes3d, p2):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = rect_to_img(corners3d.reshape(-1, 3), p2)
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)

    return boxes2d_image
