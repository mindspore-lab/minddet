import argparse
import logging as logger
import os
import random

import mindspore
import numpy as np
from det3d_ms.datasets import build_dataloader, build_dataset
from det3d_ms.models import build_detector
from det3d_ms.torchie import Config
from mindspore import context, load_checkpoint, load_param_into_net, nn
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
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--checkpoint", default="./centerpoint_ms_from_torch.ckpt", help="checkpoint"
    )
    parser.add_argument(
        "--ir_graph", default="./centerpoint_mindir_bs_4.mindir", help="IRGraph"
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
    device_idd = 6
    logger.info(f"DEVICE ID is {device_idd}")
    context.set_context(
        mode=context.GRAPH_MODE, device_id=device_idd, device_target="Ascend"
    )
    # update configs according to CLI args
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
    # set random seeds
    logger.info("Set random seed to {}".format(args.seed))
    set_random_seed(args.seed)

    net = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    logger.info("Finish build network")
    param_dict = load_checkpoint(args.checkpoint)
    logger.info(f"load_checkpoint from {args.checkpoint}")
    load_param_into_net(net, param_dict)
    net.set_train(False)
    dataset_generator = build_dataset(cfg.data.val)
    logger.info(f"test dataset length: {len(dataset_generator)}")

    workers_per_gpu = min(os.cpu_count(), cfg.data.workers_per_gpu)
    logger.info(f"workers_per_gpu: {workers_per_gpu}")
    dataset = build_dataloader(
        dataset_generator,
        4,
        workers_per_gpu,
        dist=False,
        mindrecord_dir=cfg.mindrecord_dir,
    )
    logger.info("Dataset built")
    sink_mode = False if context.get_context("mode") == context.PYNATIVE_MODE else True
    logger.info(f"sink_mode is {sink_mode}")

    graph = mindspore.load(args.ir_graph)
    logger.info(f"load IR graph from {args.ir_graph}")

    net = nn.GraphCell(graph)

    callbacks = [TimeMonitorEval(6019)]
    model = Model(
        net,
        eval_network=net,
        metrics={"mAP": MAPMetric(dataset_generator, [1, 2, 2, 1, 2, 2], cfg.work_dir)},
    )
    logger.info("start evaluating")
    result = model.eval(dataset, dataset_sink_mode=False, callbacks=callbacks)
    logger.info(f"{result}")

    # data_iterator = dataset.create_tuple_iterator(output_numpy=False, do_copy=False)
    # logger.info(f"iterator created")
    # for i, in_data in enumerate(data_iterator):
    #     output = model(*in_data)
    #     import pdb; pdb.set_trace()
    #     logger.info(f"{i}: {type(output)}")
    #     if i == 100:
    #         break


if __name__ == "__main__":
    main()
