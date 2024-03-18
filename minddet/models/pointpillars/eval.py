"""Evaluation script"""

import argparse
import os
import pickle
import warnings
from time import time

from mindspore import Tensor, context
from mindspore import dataset as de
from mindspore import load_checkpoint, load_param_into_net
from src.core.eval_utils import get_official_eval_result
from src.predict import predict, predict_kitti_to_anno
from src.utils import get_config, get_model_dataset, get_params_for_net

warnings.filterwarnings("ignore")


def run_evaluate(args):
    """run evaluate"""
    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path
    cfg = get_config(cfg_path)
    use_self_train = True
    device_id = int(os.getenv("DEVICE_ID", "0"))
    device_target = args.device_target

    context.set_context(
        mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id
    )

    model_cfg = cfg["model"]
    key_limit_range = "post_center_limit_range"
    if key_limit_range in model_cfg:
        center_limit_range = model_cfg["post_center_limit_range"]
    else:
        center_limit_range = None
    pointpillarsnet, eval_dataset, box_coder = get_model_dataset(
        cfg, is_training=False, use_self_train=use_self_train
    )

    print(pointpillarsnet)
    params = load_checkpoint(ckpt_path)
    new_params = get_params_for_net(params)
    load_param_into_net(pointpillarsnet, new_params)

    eval_input_cfg = cfg["eval_input_reader"]
    eval_column_names = eval_dataset.data_keys
    print("Info: start to load data.")
    ds = de.GeneratorDataset(
        eval_dataset,
        column_names=eval_column_names,
        python_multiprocessing=True,
        num_parallel_workers=1,
        max_rowsize=100,
        shuffle=False,
    )
    batch_size = eval_input_cfg["batch_size"]
    ds = ds.batch(batch_size, num_parallel_workers=4, drop_remainder=False)
    data_loader = ds.create_dict_iterator(output_numpy=True)

    class_names = list(eval_input_cfg["class_names"])

    gt_annos = [info["annos"] for info in eval_dataset.kitti_infos]
    dt_annos = []
    log_freq = 100
    len_dataset = len(eval_dataset)
    start = time()
    timeNet = 0
    timePredict = 0
    timeAnno = 0
    timeAll = 0
    time_data_load_all = 0
    last_iter = 0
    last_time = 0
    for i, data in enumerate(data_loader):
        start_dataload = time()
        voxels = Tensor.from_numpy(data["voxels"])
        num_points = Tensor.from_numpy(data["num_points"])
        coors = Tensor.from_numpy(data["coordinates"])
        anchors = Tensor.from_numpy(data["anchors"])
        bev_map = Tensor(data.get("bev_map", False))
        anchors_mask = Tensor.from_numpy(data["anchors_mask"])
        batch_image_shape = data["image_shape"]
        time_data_load_all += time() - start_dataload
        start1 = time()
        preds = pointpillarsnet(
            voxels, num_points, coors, anchors, anchors_mask, bev_map
        )
        timeNet += time() - start1
        start2 = time()
        preds = predict(
            preds,
            data,
            batch_image_shape,
            model_cfg,
            class_names,
            use_self_train,
            center_limit_range,
        )
        timePredict += time() - start2
        start3 = time()
        if use_self_train:
            dt_annos += predict_kitti_to_anno(
                preds, batch_image_shape, class_names, center_limit_range
            )
        else:
            dt_annos += preds
        timeAnno += time() - start3
        timeAll += time() - start_dataload
        if (
            (i % log_freq == 0 and i > 0)
            or i == 1
            or (i == len_dataset // batch_size - 1)
        ):
            time_used = time() - start
            print(
                f"processed: {i * batch_size}/{len_dataset} imgs, time elapsed: {time_used} s",
                flush=True,
            )
            print(
                f"processed: {i * batch_size}/{len_dataset} imgs, time s/img: "
                f"{(time_used - last_time) / (i - last_iter) / batch_size} s",
                flush=True,
            )
            print(
                f"processed: {i * batch_size}/{len_dataset} imgs, time Net elapsed: {timeNet} s",
                flush=True,
            )
            print(
                f"processed: {i * batch_size}/{len_dataset} imgs, time Pred elapsed: {timePredict} s",
                flush=True,
            )
            print(
                f"processed: {i * batch_size}/{len_dataset} imgs, time Anno elapsed: {timeAnno} s",
                flush=True,
            )
            print(
                f"processed: {i * batch_size}/{len_dataset} imgs, time data load all: {time_data_load_all} s",
                flush=True,
            )
            last_iter = i
            last_time = time_used
    print("Indo: infer end.")
    with open("out_gt_annos.pkl", "wb") as f:
        pickle.dump(gt_annos, f)
    with open("out_dt_annos.pkl", "wb") as f:
        pickle.dump(dt_annos, f)
    result = get_official_eval_result(
        gt_annos,
        dt_annos,
        class_names,
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True, help="Path to config file.")
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint.")
    parser.add_argument("--device_target", default="GPU", help="device target")
    parser.add_argument("--ir_graph", default="", help="IR graph.")
    parse_args = parser.parse_args()

    run_evaluate(parse_args)
