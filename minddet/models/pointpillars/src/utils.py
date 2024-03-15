"""utils"""
import yaml
from src.builder import (
    box_coder_builder,
    dataset_builder,
    model_builder,
    target_assigner_builder,
    voxel_builder,
)


def get_model_dataset(cfg, is_training=True, use_self_train=True):
    """get model dataset"""
    model_cfg = cfg["model"]
    voxel_cfg = model_cfg["voxel_generator"]
    voxel_generator = voxel_builder.build(voxel_cfg, use_self_train)

    box_coder_cfg = model_cfg["box_coder"]
    box_coder = box_coder_builder.build(box_coder_cfg)

    target_assigner_cfg = model_cfg["target_assigner"]
    target_assigner = target_assigner_builder.build(
        target_assigner_cfg,
        box_coder,
        voxel_generator.point_cloud_range,
        use_self_train,
    )

    pointpillarsnet = model_builder.build(
        model_cfg, voxel_generator, target_assigner, use_self_train
    )

    if is_training:
        input_cfg = cfg["train_input_reader"]
    else:
        input_cfg = cfg["eval_input_reader"]

    dataset = dataset_builder.build(
        input_reader_config=input_cfg,
        model_config=model_cfg,
        training=is_training,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
    )
    return pointpillarsnet, dataset, box_coder


def get_params_for_net(params):
    """get params for net"""
    new_params = {}
    for key, value in params.items():
        if key.startswith("optimizer."):
            new_params[key[10:]] = value
        elif key.startswith("network.network."):
            new_params[key[16:]] = value
    return new_params


def get_config(cfg_path):
    """get config"""
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, yaml.Loader)
    return cfg
