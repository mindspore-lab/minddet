from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )

        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def construct(
        self,
        voxels,
        coordinates,
        num_points,
        num_voxels,
        shape,
        hm_or_token=None,
        anno_box=None,
        ind=None,
        mask=None,
        cat=None,
        return_loss=True,
    ):
        # import pdb;pdb.set_trace()
        voxels = voxels.reshape((-1,) + voxels.shape[2:])
        coordinates = coordinates.reshape((-1,) + coordinates.shape[2:])
        num_points = num_points.reshape((-1,) + num_points.shape[2:])
        return_loss = self.training
        if return_loss:
            mask = self.cast(mask, mstype.uint8)
        example = {
            "voxels": voxels,
            "coordinates": coordinates,
            "num_points": num_points,
            "num_voxels": num_voxels,
            "shape": shape,
            "hm": hm_or_token,
            "anno_box": anno_box,
            "ind": ind,
            "mask": mask,
            "cat": cat,
            "token": hm_or_token,
        }

        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = {
            "features": voxels,
            "num_voxels": num_points_in_voxel,
            "coors": coordinates,
            "batch_size": batch_size,
            "input_shape": example["shape"][0],
        }

        x = self.extract_feat(data)

        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)  # 0.11s

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return (
                boxes,
                bev_feature,
                self.bbox_head.loss(example, preds, self.test_cfg),
            )
        else:
            return boxes, bev_feature, None


def convert():
    print("start convert")
    import torch

    data = torch.load("./latest.pth", map_location=torch.device("cpu"))
    keys = sorted(data["state_dict"].keys())
    key_list = []
    key_list2 = []
    for item in sorted(keys):
        if "num_batches_tracked" in item or "global_step" in item:
            continue
        if "running_mean" in item:
            key_list.append(item.replace("running_mean", "moving_mean"))
        elif "running_var" in item:
            key_list.append(item.replace("running_var", "moving_variance"))
        elif "bias" in item:
            if item.replace("bias", "running_var") in keys:
                key_list.append(item.replace("bias", "beta"))
            else:
                key_list.append(item)
        elif "weight" in item:
            if item.replace("weight", "running_var") in keys:
                key_list.append(item.replace("weight", "gamma"))
            else:
                key_list.append(item)
        else:
            key_list.append(item)
        key_list2.append(
            {"name": key_list[-1], "data": Tensor(data["state_dict"][item].numpy())}
        )
        # print(key_list[-1])
    save_checkpoint(key_list2, "./centerpoint_ms_from_torch.ckpt")


if __name__ == "__main__":
    from easydict import EasyDict

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
    # convert()
    """
    ret_dicts = model({
        "voxels": Tensor(np.zeros([batch_size1, 20, 5]), dtype=dtype),
        "coordinates": Tensor(np.zeros([batch_size1, 4]), dtype=mstype.int32),
        "num_points": Tensor(np.zeros([batch_size1]), dtype=mstype.int32),
        "num_voxels": Tensor([batch_size1], dtype=mstype.int32),
        "shape": [(512, 512, 1)],
        },
        return_loss=False)
    """
