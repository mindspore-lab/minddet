from det3d_ms.models.detectors.point_pillars import PointPillars
from mindspore import Tensor, context, load_checkpoint, load_param_into_net


def post_processing(rets, num_classes, metadata):
    # Merge branches results
    ret_list = []
    num_samples = len(rets[0])
    device = context.get_context("device_target")
    for i in range(num_samples):
        ret = {}
        if device == "Ascend":
            size = [
                (rets[j][i][1].asnumpy()[: rets[j][i][3].asnumpy()[0]] > 0).sum()
                for j in range(len(rets))
            ]
        else:
            size = [
                (rets[j][i][1].asnumpy()[: rets[j][i][3].asnumpy()] > 0).sum()
                for j in range(len(rets))
            ]
        ret["box3d_lidar"] = np.concatenate(
            [rets[j][i][0].asnumpy()[: size[j]] for j in range(len(rets))], axis=0
        )
        ret["scores"] = np.concatenate(
            [rets[j][i][1].asnumpy()[: size[j]] for j in range(len(rets))], axis=0
        )
        ret["metadata"] = metadata
        flag = 0
        data = []
        for j, num_class in enumerate(num_classes):
            data.append(rets[j][i][2].asnumpy()[: size[j]] + flag)
            flag += num_class
        ret["label_preds"] = np.concatenate(data, axis=0)
        ret_list.append(ret)
    return ret_list


if __name__ == "__main__0":
    from easydict import EasyDict
    from mindspore.common import dtype as mstype

    context.set_context(
        mode=context.PYNATIVE_MODE, device_target="Ascend"
    )  # GRAPH_MODE
    # np.set_printoptions(threshold=sys.maxsize)

    reader = EasyDict(
        {
            "type": "PillarFeatureNet",
            "num_filters": [64, 64],
            "num_input_features": 5,
            "with_distance": False,
            "voxel_size": (0.2, 0.2, 8),
            "pc_range": (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        }
    )
    backbone = EasyDict({"type": "PointPillarsScatter", "ds_factor": 1})
    neck = EasyDict(
        {
            "type": "RPN",
            "layer_nums": [3, 5, 5],
            "ds_layer_strides": [2, 2, 2],
            "ds_num_filters": [64, 128, 256],
            "us_layer_strides": [0.5, 1, 2],
            "us_num_filters": [128, 128, 128],
            "num_input_features": 64,
        }
    )
    bbox_head = EasyDict(
        {
            "type": "CenterHead",
            "in_channels": 384,
            "tasks": [
                {"num_class": 1, "class_names": ["car"]},
                {"num_class": 2, "class_names": ["truck", "construction_vehicle"]},
                {"num_class": 2, "class_names": ["bus", "trailer"]},
                {"num_class": 1, "class_names": ["barrier"]},
                {"num_class": 2, "class_names": ["motorcycle", "bicycle"]},
                {"num_class": 2, "class_names": ["pedestrian", "traffic_cone"]},
            ],
            "dataset": "nuscenes",
            "weight": 0.25,
            "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
            "common_heads": {
                "reg": (2, 2),
                "height": (1, 2),
                "dim": (3, 2),
                "rot": (2, 2),
                "vel": (2, 2),
            },
        }
    )
    test_cfg = EasyDict(
        {
            "circular_nms": True,
            "min_radius": [4, 12, 10, 1, 0.85, 0.175],
            "post_center_limit_range": Tensor([-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]),
            "max_per_img": 500,
            "nms": {
                "nms_pre_max_size": 1000,
                "nms_post_max_size": 83,
                "nms_iou_threshold": 0.2,
            },
            "score_threshold": 0.1,
            "pc_range": [-51.2, -51.2],
            "out_size_factor": 4,
            "voxel_size": [0.2, 0.2],
        }
    )
    model = PointPillars(
        reader=reader,
        backbone=backbone,
        neck=neck,
        bbox_head=bbox_head,
        test_cfg=EasyDict(test_cfg),
    )
    param_dict = load_checkpoint("centerpoint_ms.ckpt")
    load_param_into_net(model, param_dict)
    model.set_train(False)
    dtype = mstype.float32
    model = model.to_float(dtype)
    batch_size1 = 7988
    import pickle

    data = pickle.load(open("example_np.pkl", "rb"))

    ret_dicts = model(
        {
            "voxels": Tensor(data["voxels"], dtype=dtype),
            "coordinates": Tensor(data["coordinates"], dtype=mstype.int32),
            "num_points": Tensor(data["num_points"], dtype=mstype.int32),
            "num_voxels": Tensor(data["num_voxels"], dtype=mstype.int32),
            "shape": [(512, 512, 1)],
        },
        return_loss=False,
    )
    print(
        post_processing(
            ret_dicts,
            model.bbox_head.num_classes,
            metadata={
                "num_point_features": 5,
                "token": "1dba6570f0774b5cb87de6694bb338c0",
            },
        )
    )


if __name__ == "__main__":
    import numpy as np

    # context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend") # GRAPH_MODE
    # np.set_printoptions(threshold=sys.maxsize)
    from easydict import EasyDict
    from mindspore.common import dtype as mstype

    reader = EasyDict(
        {
            "type": "PillarFeatureNet",
            "num_filters": [64, 64],
            "num_input_features": 5,
            "with_distance": False,
            "voxel_size": (0.2, 0.2, 8),
            "pc_range": (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        }
    )
    backbone = EasyDict({"type": "PointPillarsScatter", "ds_factor": 1})
    neck = EasyDict(
        {
            "type": "RPN",
            "layer_nums": [3, 5, 5],
            "ds_layer_strides": [2, 2, 2],
            "ds_num_filters": [64, 128, 256],
            "us_layer_strides": [0.5, 1, 2],
            "us_num_filters": [128, 128, 128],
            "num_input_features": 64,
        }
    )
    bbox_head = EasyDict(
        {
            "type": "CenterHead",
            "in_channels": 384,
            "tasks": [
                {"num_class": 1, "class_names": ["car"]},
                {"num_class": 2, "class_names": ["truck", "construction_vehicle"]},
                {"num_class": 2, "class_names": ["bus", "trailer"]},
                {"num_class": 1, "class_names": ["barrier"]},
                {"num_class": 2, "class_names": ["motorcycle", "bicycle"]},
                {"num_class": 2, "class_names": ["pedestrian", "traffic_cone"]},
            ],
            "dataset": "nuscenes",
            "weight": 0.25,
            "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
            "common_heads": {
                "reg": (2, 2),
                "height": (1, 2),
                "dim": (3, 2),
                "rot": (2, 2),
                "vel": (2, 2),
            },
        }
    )
    test_cfg = EasyDict(
        {
            "circular_nms": True,
            "min_radius": [4, 12, 10, 1, 0.85, 0.175],
            "post_center_limit_range": Tensor([-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]),
            "max_per_img": 500,
            "nms": {
                "nms_pre_max_size": 1000,
                "nms_post_max_size": 83,
                "nms_iou_threshold": 0.2,
            },
            "score_threshold": 0.1,
            "pc_range": [-51.2, -51.2],
            "out_size_factor": 4,
            "voxel_size": [0.2, 0.2],
        }
    )
    model = PointPillars(
        reader=reader,
        backbone=backbone,
        neck=neck,
        bbox_head=bbox_head,
        test_cfg=EasyDict(test_cfg),
    )
    param_dict = load_checkpoint("centerpoint_ms.ckpt")
    load_param_into_net(model, param_dict)
    model.set_train(False)
    dtype = mstype.float32
    model = model.to_float(dtype)
    batch_size1 = 7988
    import pickle

    data_dump = pickle.load(open("data_dump.pkl", "rb"))
    ret_dump = []
    for data in data_dump:
        ret_dicts = model(
            **{
                "voxels": Tensor(data["voxels"].numpy(), dtype=dtype),
                "coordinates": Tensor(data["coordinates"].numpy(), dtype=mstype.int32),
                "num_points": Tensor(data["num_points"].numpy(), dtype=mstype.int32),
                "num_voxels": Tensor(data["num_voxels"].numpy(), dtype=mstype.int32),
                "shape": [(512, 512, 1)],
            },
            return_loss=False
        )
        ret = post_processing(ret_dicts, model.bbox_head.num_classes, data["metadata"])
        ret_dump.append(ret)
        print(ret)
    pickle.dump(ret_dump, open("ret_dump_one.pkl", "wb"))
