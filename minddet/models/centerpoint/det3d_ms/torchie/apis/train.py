from __future__ import division

from collections import OrderedDict

import torch
from mindspore import Tensor


def example_to_device(example, device=None, non_blocking=False) -> dict:
    example_torch = {}
    for k, v in example.items():
        if k in [
            "anchors",
            "anchors_mask",
            "reg_targets",
            "reg_weights",
            "labels",
            "points",
        ]:
            example_torch[k] = [Tensor(res.cpu().numpy()) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points",
        ]:
            example_torch[k] = v
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = Tensor(v1.cpu().numpy())
            example_torch[k] = calib
        else:
            example_torch[k] = v
        if type(example_torch[k]) == torch.Tensor:
            example_torch[k] = Tensor(example_torch[k].cpu().numpy())
        if k == "shape":
            example_torch[k] = tuple(example_torch[k].tolist())

    return example_torch


def parse_second_losses(losses):
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


def batch_processor(model, data, train_mode, **kwargs):
    device = None

    # data = example_convert_to_torch(data, device=device)
    example = example_to_device(data, device, non_blocking=False)

    del data

    if train_mode:
        losses = model(example, return_loss=True)
        loss, log_vars = parse_second_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0])
        )
        return outputs
    else:
        return model(example, return_loss=False)


def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]


def build_one_cycle_optimizer(model, optimizer_config):
    optimizer = None
    return optimizer
