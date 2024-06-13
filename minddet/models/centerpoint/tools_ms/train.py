import argparse
import logging as logger
import math
import os
import random

import mindspore
import mindspore.dataset as ds
import numpy as np
from det3d_ms.datasets import build_dataloader, build_dataset
from det3d_ms.models import build_detector
from det3d_ms.torchie import Config
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.model import Model
from tools_ms.optim_zy import AdamW
from tools_ms.utils.utils import TimeMonitor, TrainOneStepCellWrapper

logger.basicConfig(level=logger.INFO)
cur_path = os.path.split(os.path.realpath(__file__))[0] + "/../"
os.chdir(cur_path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    ds.config.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", default=None, help="train config file path")
    parser.add_argument("--data_url", default="/nuScenes", help="data_url")
    parser.add_argument("--train_url", default="./work_dirs/debug", help="output url")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
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
    parser.add_argument(
        "--is_dynamic_loss_scale",
        type=int,
        default=0,
        help="is_dynamic_loss_scale",
    )
    parser.add_argument(
        "--is_dump",
        action="store_true",
        help="is_dump",
    )
    parser.add_argument("--epochs", type=int, default=20, help="total epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning_rate"
    )
    parser.add_argument("--div_factor", type=float, default=10.0, help="div_factor")
    parser.add_argument("--pct_start", type=float, default=0.4, help="pct_start")
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


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_cosine_annealing_lr(
    lr_max, lr_min, steps_per_epoch, warmup_steps, max_epoch, div_factor
):
    warmup_init_lr = lr_max / div_factor
    total_steps = int(max_epoch * steps_per_epoch)
    decay_steps = total_steps - warmup_steps

    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, lr_max, warmup_init_lr)
        else:
            lr = (
                lr_min
                + (lr_max - lr_min)
                * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
                / 2
            )
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def main():
    args = parse_args()
    if args.is_dump:
        logger.info("inside dump")  # TODO
        save_graphs_path = os.path.join(args.train_url, "lr")
        dump_url = os.path.join(args.train_url, "dump")
        os.makedirs(save_graphs_path, exist_ok=True)
        os.makedirs(dump_url, exist_ok=True)
        context.set_context(
            save_graphs=True,
            save_graphs_path=save_graphs_path,
            reserve_class_name_in_scope=False,
        )
        os.environ["MS_DIAGNOSTIC_DATA_PATH"] = dump_url
        os.environ["MINDSPORE_DUMP_CONFIG"] = cur_path + "/dump.json"

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)
    rank_id = 0
    args.gpus = int(os.getenv("RANK_SIZE", "1"))
    distributed = args.gpus > 1
    logger.info("Distributed training: {}".format(distributed))

    rank_id = int(os.getenv("RANK_ID", "0"))
    os.makedirs(cur_path + "data", exist_ok=True)
    print("ln -s " + args.data_url + " " + cur_path + "data/nuScenes")
    if not os.path.exists(cur_path + "data/nuScenes"):
        os.system("ln -s " + args.data_url + " " + cur_path + "data/nuScenes")
    if rank_id == 0:
        logger.info(f"cur_path: {cur_path}")
        logger.info(f"ls -al {args.data_url}")
        os.system("ls -al " + args.data_url)
        logger.info("ls -al " + cur_path + "data")
        os.system("ls -al " + cur_path + "data")
        logger.info("ls -al " + cur_path + "data/nuScenes")
        os.system("ls -al " + cur_path + "data/nuScenes")
    device_id = int(os.getenv("DEVICE_ID", "0"))
    logger.info(f"device_id: {device_id}, gpus: {args.gpus}, rank_id: {rank_id}")
    if distributed:
        logger.info("Distributed running!")  # TODO
        context.set_context(
            mode=context.GRAPH_MODE, device_id=device_id, device_target="Ascend"
        )
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            device_num=args.gpus,
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
        )
        logger.info(f"Device number: {args.gpus}")
    else:
        logger.info("NOT Distributed")  # TODOsss
        context.set_context(
            mode=context.GRAPH_MODE, device_id=device_id, device_target="Ascend"
        )
    if mindspore.get_context("device_target") == "GPU":
        logger.info("NOT Distributed but GPU")  # TODO
        context.set_context(enable_graph_kernel=True)
    args.config = (
        cur_path + "configs_ms/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py"
    )
    cfg = Config.fromfile(args.config)
    logger.info(f"epoch: {args.epochs}")
    cfg.local_rank = args.local_rank
    cfg.gpus = args.gpus
    cfg.lr_config.lr_max = args.learning_rate
    logger.info(f"lr_max: {args.learning_rate}")
    cfg.lr_config.div_factor = args.div_factor
    cfg.lr_config.pct_start = args.pct_start
    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus
    net = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if args.checkpoint:
        param_dict = load_checkpoint(args.checkpoint)
        load_param_into_net(net, param_dict)
        logger.info(f"load_checkpoint from {args.checkpoint}")
    logger.info(f"pwd3: {os.getcwd()}")
    logger.info(f"cfg: {cfg}")
    logger.info(f"args: {args}")
    logger.info(f"os.environ: {os.environ}")

    dataset_generator = build_dataset(cfg.data.train)
    dataset = build_dataloader(
        dataset_generator,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        num_devices=args.gpus,
        rank_id=rank_id,
        dist=distributed,
        mindrecord_dir=cfg.train_mindrecord_dir,
    )

    total_step = dataset.get_dataset_size() * cfg.total_epochs
    logger.info(
        f"dataset_generator: {len(dataset_generator)}, steps per epoch: {dataset.get_dataset_size()}, "
        f"total_step: {total_step}"
    )

    lr_max = 0.002
    lr_min = 1e-06

    steps_per_epoch = dataset.get_dataset_size()
    warmup_steps = steps_per_epoch * 2.5
    lr_schedule = warmup_cosine_annealing_lr(
        lr_max, lr_min, steps_per_epoch, warmup_steps, 20, 10
    )

    lr_schedule = lr_schedule * 10

    logger.info("Cosine Annealing")
    logger.info(
        f"learning rate schedule: {lr_schedule}, lr_max: {cfg.lr_config.lr_max}, "
        f"div_factor: {cfg.lr_config.div_factor}, pct_start: {cfg.lr_config.pct_start}, "
        f"total_step: {total_step}"
    )

    opt = AdamW(
        params=net.trainable_params(),
        learning_rate=Tensor(lr_schedule),
        beta1=0.95,
        beta2=0.999,
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    net.set_train()
    if args.is_dynamic_loss_scale:
        logger.info(f"is_dynamic_loss_scale: {args.is_dynamic_loss_scale}")
        loss_scale_manager = DynamicLossScaleManager(
            init_loss_scale=args.is_dynamic_loss_scale,
            scale_factor=2,
            scale_window=2000,
        )
        model = Model(net, optimizer=opt, loss_scale_manager=loss_scale_manager)
    else:
        logger.info("not dynamic loss scale")  # TODO
        model = Model(net, optimizer=opt)

    callbacks = [LossMonitor(1), TimeMonitor(dataset.get_dataset_size())]
    logger.info("define callbacks")  # TODO
    if rank_id == 0:
        logger.info("rank_id == 0")  # TODO
        config_ck = CheckpointConfig(
            save_checkpoint_steps=dataset.get_dataset_size(),
            keep_checkpoint_max=cfg.total_epochs,
        )
        logger.info("checkpoint config")  # TODO
        os.makedirs(args.train_url, exist_ok=True)
        ckpt_callback = ModelCheckpoint(
            "centerpoint", directory=args.train_url, config=config_ck
        )
        callbacks += [ckpt_callback]
    logger.info("before train")  # TODO

    network = TrainOneStepCellWrapper(net, opt)
    model = Model(network=network)
    model.train(
        epoch=cfg.total_epochs,
        dataset_sink_mode=False,
        train_dataset=dataset,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
