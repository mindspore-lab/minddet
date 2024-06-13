import numpy as np
from det3d_ms.models import build_detector
from det3d_ms.torchie import Config
from mindspore import Tensor, context, export, load_checkpoint


def main():
    config_path = "configs_ms/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py"
    cfg = Config.fromfile(config_path)
    context.set_context(mode=context.GRAPH_MODE, device_id=6, device_target="Ascend")

    center_point = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg
    )
    load_checkpoint("./centerpoint_ms.ckpt", net=center_point)

    voxel = np.random.uniform(0.0, 1.0, size=[4, 60000, 20, 5]).astype(np.float32)
    coordinates = np.random.uniform(0.0, 1.0, size=[4, 60000, 4]).astype(np.int32)
    num_points = np.random.uniform(0.0, 1.0, size=[4, 60000]).astype(np.int32)
    num_voxels = np.random.uniform(0.0, 1.0, size=[4, 1]).astype(np.int64)
    shape = np.random.uniform(0.0, 1.0, size=[4, 3]).astype(np.int64)
    hm_or_token = np.random.uniform(0.0, 1.0, size=[4, 32]).astype(np.int64)

    export(
        center_point,
        Tensor(voxel),
        Tensor(coordinates),
        Tensor(num_points),
        Tensor(num_voxels),
        Tensor(shape),
        Tensor(hm_or_token),
        file_name="centerpoint_mindir_bs_4",
        file_format="MINDIR",
    )


if __name__ == "__main__":
    main()
