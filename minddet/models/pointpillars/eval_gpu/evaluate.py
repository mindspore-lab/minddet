import os
import pickle
import sys
import time

import fire

# print(sys.path)
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
print(sys.path)

import kitti_common as kitti
from eval import get_coco_eval_result, get_official_eval_result


def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path, result_path, current_class=0):
    with open(label_path, "rb") as f:
        gt_annos = pickle.load(f)
    with open(result_path, "rb") as f:
        dt_annos = pickle.load(f)
    result = get_official_eval_result(gt_annos, dt_annos, current_class)
    print(result[0])
    return
    # dt_annos = kitti.get_label_annos(result_path)
    # if score_thresh > 0:
    #     dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    # val_image_ids = _read_imageset_file(label_split_file)
    # gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    # if coco:
    #     return get_coco_eval_result(gt_annos, dt_annos, current_class)
    # else:


if __name__ == "__main__":
    fire.Fire()
