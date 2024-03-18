"""dataset"""
import pickle
from functools import partial

from src.core import box_np_ops
from src.data.preprocess import _read_and_prep


class KittiDataset:
    """Kitti dataset"""

    def __init__(
        self,
        info_path,
        root_path,
        num_point_features,
        target_assigner,
        feature_map_size,
        prep_func,
        data_keys,
    ):
        with open(info_path, "rb") as f:
            infos = pickle.load(f)
        self._root_path = root_path
        self._kitti_infos = infos
        self._num_point_features = num_point_features
        # generate anchors cache
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)
        self.data_keys = data_keys

    def __len__(self):
        return len(self._kitti_infos)

    @property
    def kitti_infos(self):
        """kitti infos"""
        return self._kitti_infos

    def __getitem__(self, idx):
        example = _read_and_prep(
            info=self._kitti_infos[idx],
            root_path=self._root_path,
            num_point_features=self._num_point_features,
            prep_func=self._prep_func,
        )
        data = [example[key] for key in self.data_keys]
        return tuple(data)
