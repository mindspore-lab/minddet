import logging as logger
import os

import mindspore.dataset as ds
import numpy as np
from det3d_ms.datasets import build_dataset
from det3d_ms.datasets.loader.sampler import GroupSampler
from det3d_ms.torchie import Config
from mindspore.mindrecord import FileWriter

logger.basicConfig(level=logger.INFO)
cur_path = os.path.split(os.path.realpath(__file__))[0] + "/../"
os.chdir(cur_path)


def build_mindrecord(dataset, num_workers, output_path, test_mode):
    if test_mode:
        sampler = GroupSampler(dataset, 1, shuffle=False)
        print("==========build test mindrecord==========")
        dataset = ds.GeneratorDataset(
            dataset,
            ["voxels", "coordinates", "num_points", "num_voxels", "shape", "token"],
            shuffle=False,
            sampler=sampler,
            num_parallel_workers=num_workers,
            max_rowsize=24,
        )
        writer = FileWriter(file_name=output_path, shard_num=20, overwrite=True)
        writer.set_page_size(1 << 25)
        schema_json = {
            "voxels": {"type": "float32", "shape": [60000, 20, 5]},
            "coordinates": {"type": "int32", "shape": [60000, 3]},
            "num_points": {"type": "int32", "shape": [60000]},
            "num_voxels": {"type": "int64", "shape": [1]},
            "shape": {"type": "int64", "shape": [3]},
            "token": {"type": "int64", "shape": [32]},
        }
        writer.add_schema(schema_json, "pc_schema")

        ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
        dataset_size = dataset.get_dataset_size()

        data = []
        for i, item in enumerate(ds_iter):
            data.append(item)
            if i % 200 == 0:
                writer.write_raw_data(data, parallel_writer=False)
                data = []
                print(f"=====progress{i}/{dataset_size}")
        if data:
            writer.write_raw_data(data, parallel_writer=False)
            data = None
        writer.commit()
    else:
        print("==========build train mindrecord==========")
        dataset = ds.GeneratorDataset(
            dataset,
            [
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
            shuffle=False,
            sampler=None,
            num_parallel_workers=num_workers,
            max_rowsize=24,
        )
        writer = FileWriter(file_name=output_path, shard_num=100, overwrite=True)
        writer.set_page_size(1 << 24)
        schema_json = {
            "voxels": {"type": "float32", "shape": [30000, 20, 5]},
            "coordinates": {"type": "int32", "shape": [30000, 3]},
            "num_points": {"type": "int32", "shape": [30000]},
            "num_voxels": {"type": "int64", "shape": [1]},
            "shape": {"type": "int64", "shape": [3]},
            "hm": {"type": "float32", "shape": [6, 2, 128, 128]},
            "anno_box": {"type": "float32", "shape": [6, 500, 10]},
            "ind": {"type": "int64", "shape": [6, 500]},
            "mask": {"type": "int32", "shape": [6, 500]},
            "cat": {"type": "int64", "shape": [6, 500]},
        }
        writer.add_schema(schema_json, "pc_schema")

        ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
        dataset_size = dataset.get_dataset_size()

        data = []
        for i, item in enumerate(ds_iter):
            item["mask"] = item["mask"].astype(np.int32)
            data.append(item)
            if i % 200 == 0:
                writer.write_raw_data(data, parallel_writer=False)
                data = []
                print(f"=====progress{i}/{dataset_size}")
        if data:
            writer.write_raw_data(data, parallel_writer=False)
            data = None
        writer.commit()


def main(train_output_path, test_output_path):
    config_path = (
        cur_path + "configs_ms/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py"
    )
    cfg = Config.fromfile(config_path)
    train_dataset_generator = build_dataset(cfg.data.train)
    test_dataset_generator = build_dataset(cfg.data.val)
    # build train mindrecord
    build_mindrecord(train_dataset_generator, 10, train_output_path, False)
    # build test mindrecord
    build_mindrecord(test_dataset_generator, 10, test_output_path, True)


if __name__ == "__main__":
    train_output_path = ""
    test_output_path = ""
    main(train_output_path, test_output_path)
