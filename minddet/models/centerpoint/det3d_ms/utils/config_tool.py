# This file contains some config modification function.
# some functions should be only used for KITTI dataset.
import numpy as np


def get_downsample_factor(model_config):
    try:
        neck_cfg = model_config["neck"]
    except ValueError:
        model_config = model_config["first_stage_cfg"]
        neck_cfg = model_config["neck"]
    downsample_factor = np.prod(neck_cfg.get("ds_layer_strides", [1]))
    if len(neck_cfg.get("us_layer_strides", [])) > 0:
        downsample_factor /= neck_cfg.get("us_layer_strides", [])[-1]

    backbone_cfg = model_config["backbone"]
    downsample_factor *= backbone_cfg["ds_factor"]
    downsample_factor = int(downsample_factor)
    assert downsample_factor > 0
    return downsample_factor
