import platform

import mindspore.dataset as ds
import numpy as np

from .sampler import DistributedGroupSampler, GroupSampler

if platform.system() != "Windows":
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def collate_kitti(coordinates, batchInfo):
    coors = []
    for i, coor in enumerate(coordinates):
        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
        coors.append(coor_pad)
    return (coors,)


def build_dataloader(
    dataset,
    batch_size,
    workers_per_gpu,
    num_devices=1,
    rank_id=0,
    dist=True,
    mindrecord_dir=None,
    **kwargs
):
    num_workers = workers_per_gpu
    print("=====" * 50, num_workers, flush=True)
    if dataset.test_mode:
        print("==========build test dataloader==========")
        sampler = GroupSampler(dataset, batch_size, shuffle=False)
        dataset = ds.MindDataset(
            mindrecord_dir,
            shuffle=False,
            num_parallel_workers=8,
            num_shards=num_devices,
            shard_id=rank_id,
            columns_list=[
                "voxels",
                "coordinates",
                "num_points",
                "num_voxels",
                "shape",
                "token",
            ],
        )

        data_loader = dataset.batch(
            batch_size=batch_size,
            per_batch_map=collate_kitti,
            input_columns=["coordinates"],
            num_parallel_workers=num_workers,
            drop_remainder=False,
        )
    else:
        num_devices, rank_id = (None, None) if not dist else (num_devices, rank_id)
        sampler = (
            DistributedGroupSampler(dataset, batch_size, num_devices, rank_id)
            if dist
            else None
        )
        # sampler = None
        if sampler:
            print("==========build train distributed data loader==========")
            dataset = ds.MindDataset(
                mindrecord_dir,
                shuffle=True,
                num_parallel_workers=8,
                num_shards=num_devices,
                shard_id=rank_id,
                columns_list=[
                    "voxels",
                    "coordinates",
                    "num_points",
                    "num_voxels",
                    "shape",
                    "hm",
                    "anno_box",
                    "ind",
                    "mask",
                    "cat",
                ],
            )
        else:
            print("==========build train non-distributed data loader==========")
            dataset = ds.MindDataset(
                mindrecord_dir,
                shuffle=True,
                num_parallel_workers=8,
                num_shards=num_devices,
                shard_id=rank_id,
                columns_list=[
                    "voxels",
                    "coordinates",
                    "num_points",
                    "num_voxels",
                    "shape",
                    "hm",
                    "anno_box",
                    "ind",
                    "mask",
                    "cat",
                ],
            )

        data_loader = dataset.batch(
            batch_size=batch_size,
            per_batch_map=collate_kitti,
            input_columns=["coordinates"],
            num_parallel_workers=3,
            drop_remainder=True,
        )  # , python_multiprocessing=True)

    return data_loader
