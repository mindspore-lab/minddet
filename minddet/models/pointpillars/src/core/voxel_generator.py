"""voxel generator"""
import numpy as np
from src.core.point_cloud.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    """voxel generator"""

    def __init__(self, voxel_size, point_cloud_range, max_num_points):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._grid_size = grid_size

    def generate(self, points, max_voxels):
        """generate"""
        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            max_voxels,
        )

    @property
    def voxel_size(self):
        """voxel size"""
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        """max num points per voxel"""
        return self._max_num_points

    @property
    def point_cloud_range(self):
        """point cloud range"""
        return self._point_cloud_range

    @property
    def grid_size(self):
        """grid size"""
        return self._grid_size
