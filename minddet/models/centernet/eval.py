"""
CenterNet evaluation script.
"""

import copy
import json
import os
import time

import cv2
import mindspore.log as logger
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src import (
    CenterNetDetEval,
    convert_eval_format,
    merge_outputs,
    post_process,
    visual_image,
)
from src.dataset import COCOHP
from src.model_utils.config import config, dataset_config, eval_config, net_config
from src.model_utils.moxing_adapter import moxing_wrapper

_current_dir = os.path.dirname(os.path.realpath(__file__))


def modelarts_pre_process():
    """modelarts pre process function."""
    try:
        from nms import soft_nms

        print("soft_nms_attributes: {}".format(soft_nms.__dir__()))
    except ImportError:
        print("NMS not installed! trying installing...\n")
        cur_path = os.path.dirname(os.path.abspath(__file__))
        os.system(
            "cd {}/CenterNet/src/lib/external/ && make && python setup.py install && cd - ".format(
                cur_path
            )
        )
        try:
            from nms import soft_nms

            print("soft_nms_attributes: {}".format(soft_nms.__dir__()))
        except ImportError:
            print('Installing failed! check if the folder "./CenterNet" exists.')
        else:
            print("Install nms successfully")
    config.data_dir = config.data_path
    config.load_checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), config.load_checkpoint_path
    )


@moxing_wrapper(pre_process=modelarts_pre_process)
def predict():
    """
    Predict function
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)
        enable_nms_fp16 = False
    else:
        enable_nms_fp16 = True
    config.run_mode = "val"
    logger.info("Begin creating {} dataset".format(config.run_mode))
    coco = COCOHP(
        dataset_config,
        run_mode=config.run_mode,
        net_opt=net_config,
        enable_visual_image=config.visual_image,
        save_path=config.save_result_dir,
    )
    coco.init(config.data_dir, keep_res=eval_config.keep_res)
    dataset = coco.create_eval_dataset()

    net_for_eval = CenterNetDetEval(net_config, eval_config.K, enable_nms_fp16)
    net_for_eval.set_train(False)
    param_dict = load_checkpoint(config.load_checkpoint_path)
    param_not_load, ckpt_not_load = load_param_into_net(
        net_for_eval.network, param_dict
    )
    print("param not load", param_not_load)

    save_path = os.path.join(config.save_result_dir, config.run_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if config.visual_image == "true":
        save_pred_image_path = os.path.join(save_path, "pred_image")
        if not os.path.exists(save_pred_image_path):
            os.makedirs(save_pred_image_path)
        save_gt_image_path = os.path.join(save_path, "gt_image")
        if not os.path.exists(save_gt_image_path):
            os.makedirs(save_gt_image_path)

    total_nums = dataset.get_dataset_size()
    print("\n========================================\n")
    print("Total images num: ", total_nums)
    print("Processing, please wait a moment.")

    pred_annos = {"images": [], "annotations": []}

    index = 0
    iterator = dataset.create_dict_iterator(num_epochs=1)
    for data in iterator:
        index += 1
        image = data["image"]
        image_id = data["image_id"].asnumpy().reshape((-1))[0]

        # run prediction
        detections = []
        for scale in eval_config.multi_scales:
            images, meta = coco.pre_process_for_test(image.asnumpy(), scale)
            start = time.time()
            detection = net_for_eval(Tensor(images))

            dets = post_process(
                detection.asnumpy(), meta, scale, dataset_config.num_classes
            )
            end = time.time()
            print(
                "Image {}/{} id: {} cost time {} ms".format(
                    index, total_nums, image_id, (end - start) * 1000.0
                )
            )
            detections.append(dets)

        # post-process
        detections = merge_outputs(
            detections, dataset_config.num_classes, eval_config.SOFT_NMS
        )
        # get prediction result
        pred_json = convert_eval_format(detections, image_id, eval_config.valid_ids)
        gt_image_info = coco.coco.loadImgs([image_id])

        for image_info in pred_json["images"]:
            pred_annos["images"].append(image_info)
        for image_anno in pred_json["annotations"]:
            pred_annos["annotations"].append(image_anno)
        if config.visual_image == "true":
            img_file = os.path.join(coco.image_path, gt_image_info[0]["file_name"])
            gt_image = cv2.imread(img_file)
            if config.run_mode != "test":
                annos = coco.coco.loadAnns(coco.anns[image_id])
                visual_image(
                    copy.deepcopy(gt_image),
                    annos,
                    save_gt_image_path,
                    score_threshold=eval_config.score_thresh,
                )
            anno = copy.deepcopy(pred_json["annotations"])
            visual_image(
                gt_image,
                anno,
                save_pred_image_path,
                score_threshold=eval_config.score_thresh,
            )

    # save results
    save_path = os.path.join(config.save_result_dir, config.run_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pred_anno_file = os.path.join(save_path, "{}_pred_result.json").format(
        config.run_mode
    )
    json.dump(pred_annos, open(pred_anno_file, "w"))
    pred_res_file = os.path.join(save_path, "{}_pred_eval.json").format(config.run_mode)
    json.dump(pred_annos["annotations"], open(pred_res_file, "w"))
    # pred_res_file = '/disk1/guoshipeng/models/research/cv/centernet_resnet50_v1/val/val_pred_eval.json'
    if config.run_mode != "test" and config.enable_eval:
        run_eval(coco.annot_path, pred_res_file)


def run_eval(gt_anno, pred_anno):
    """evaluation by coco api"""
    coco = COCO(gt_anno)
    coco_dets = coco.loadRes(pred_anno)
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    # coco_eval.params.maxDets = list((100, 300, 1000))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    predict()
