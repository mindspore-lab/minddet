"""Parse arguments"""

import argparse
import ast
import os
from pprint import pformat

import numpy as np
import yaml


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, str) and (
                v[:9] == "np.array(" and v[-17:] == "dtype=np.float32)"
            ):
                v = np.array(
                    ast.literal_eval(v[9 : v.rfind("]") + 1]), dtype=np.float32
                )
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_cli_to_yaml(
    parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"
):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(
        description="[REPLACE THIS at config.py]", parents=[parser]
    )
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = (
                helper[item]
                if item in helper
                else "Please reference to {}".format(cfg_path)
            )
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument(
                    "--" + item,
                    type=ast.literal_eval,
                    default=cfg[item],
                    choices=choice,
                    help=help_description,
                )
            else:
                parser.add_argument(
                    "--" + item,
                    type=type(cfg[item]),
                    default=cfg[item],
                    choices=choice,
                    help=help_description,
                )
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, "r") as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError(
                    "At most 3 docs (config, description for help, choices) are supported in config yaml"
                )
            print(cfg_helper)
        except ValueError:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def extra_operations(cfg):
    """
    Do extra work on Config object.

    Args:
        cfg: Object after instantiation of class 'Config'.
    """
    cfg.train_config.Adam.decay_filter = (
        lambda x: x.name.endswith(".bias")
        or x.name.endswith(".beta")
        or x.name.endswith(".gamma")
    )
    cfg.export_config.input_res = cfg.dataset_config.input_res_test
    if cfg.export_load_ckpt:
        cfg.export_config.ckpt_file = cfg.export_load_ckpt
    if cfg.export_name:
        cfg.export_config.export_name = cfg.export_name
    if cfg.export_format:
        cfg.export_config.export_format = cfg.export_format


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(current_dir, "../../default_config.yaml"),
        help="Config file path",
    )
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(
        parser=parser,
        cfg=default,
        helper=helper,
        choices=choices,
        cfg_path=path_args.config_path,
    )
    final_config = merge(args, default)
    config_obj = Config(final_config)
    extra_operations(config_obj)
    return config_obj


config = get_config()
dataset_config = config.dataset_config
net_config = config.net_config
train_config = config.train_config
eval_config = config.eval_config
export_config = config.export_config
