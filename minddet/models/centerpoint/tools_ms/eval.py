import argparse
import logging as logger
import os
import random

import mindspore
import numpy as np
from det3d_ms.datasets import build_dataloader, build_dataset
from det3d_ms.models import build_detector
from det3d_ms.torchie import Config
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.nn.metrics import Metric
from mindspore.train.model import Model

from .utils.utils import TimeMonitorEval

# ModelCheckpoint, TimeMonitor)
# from mindspore.common.api import _cell_graph_executor
# _cell_graph_executor.set_jit_config(jit_config={"jit_level": "o0"})

logger.basicConfig(
    level=logger.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", default=None, help="train config file path")
    parser.add_argument(
        "--work_dir", default="work_dir", help="the dir to save logs and models"
    )
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--checkpoint", default=None, help="checkpoint")
    parser.add_argument(
        "--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


class MAPMetric(Metric):
    """AUC metric for AutoDis model."""

    def __init__(self, dataset, num_classes, work_dir):
        super(MAPMetric, self).__init__()
        self.pred_probs = []
        self.true_labels = []
        self.predictions = {}
        self.num_classes = num_classes
        self.dataset = dataset
        self.device = context.get_context("device_target")
        self.work_dir = work_dir

    def post_processing(self, rets, token):
        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        for i in range(num_samples):
            ret = {}
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
            ret["metadata"] = {
                "token": "".join([chr(x) for x in token[i]])
            }  # token[i]}
            flag = 0
            data = []
            for j, num_class in enumerate(self.num_classes):
                data.append(rets[j][i][2].asnumpy()[: size[j]] + flag)
                flag += num_class
            ret["label_preds"] = np.concatenate(data, axis=0)
            ret_list.append(ret)
        return ret_list

    def clear(self):
        """Clear the internal evaluation result."""
        self.pred_probs = []
        self.true_labels = []
        self.predictions = {}

    def update(self, *inputs):
        outputs = self.post_processing(inputs[0], inputs[1].asnumpy())
        for output in outputs:
            token = output["metadata"]["token"]
            self.predictions.update({token: output})
        logger.info(f"{len(self.predictions)}")

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError("true_labels.size() is not equal to pred_probs.size()")
        map = self.dataset.evaluation(self.predictions, self.work_dir, False)
        return map


def main():
    args = parse_args()
    args.config = "configs_ms/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py"
    cfg = Config.fromfile(args.config)
    device_id = int(os.getenv("DEVICE_ID", "0"))
    logger.info(f"DEVICE ID is {device_id}")
    context.set_context(
        mode=context.GRAPH_MODE, device_id=device_id, device_target="Ascend"
    )
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    distributed = False

    cfg.local_rank = args.local_rank
    cfg.gpus = args.gpus
    cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus
    args.rank = 0
    args.device_num = 0
    # init logger before other steps
    logger.info("Distributed training: {}".format(distributed))

    if args.local_rank == 0:
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(cfg.work_dir, exist_ok=True)
    logger.info(f"working directory: {cfg.work_dir}")
    # set random seeds
    logger.info("Set random seed to {}".format(args.seed))
    set_random_seed(args.seed)

    net = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    logger.info("Finish build network")  # todo
    param_dict = load_checkpoint(args.checkpoint)
    logger.info(f"load_checkpoint from {args.checkpoint}")
    load_param_into_net(net, param_dict)
    net.set_train(False)
    dataset_generator = build_dataset(cfg.data.val)  # NuScenesDataset
    logger.info(f"test dataset length: {len(dataset_generator)}")
    dataset = build_dataloader(
        dataset_generator,
        1,
        min(os.cpu_count(), cfg.data.workers_per_gpu),
        dist=False,
        mindrecord_dir=cfg.test_mindrecord_dir,
    )
    logger.info("Dataset built")  # todo
    sink_mode = False if context.get_context("mode") == context.PYNATIVE_MODE else True
    logger.info(f"sink_mode is {sink_mode}")
    callbacks = [TimeMonitorEval(6019)]  # TODO 确定一下怎么把dataset的长度传进callbacks里
    model = Model(
        net,
        eval_network=net,
        metrics={
            "mAP": MAPMetric(dataset_generator, net.bbox_head.num_classes, cfg.work_dir)
        },
    )
    logger.info("start evaluating")
    result = model.eval(dataset, dataset_sink_mode=sink_mode, callbacks=callbacks)
    logger.info(f"{result}")


if __name__ == "__main__":
    main()
