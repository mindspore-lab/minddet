import mindspore
import torch
from src.centernet_det import CenterNetLossCell
from src.model_utils.config import net_config
import numpy as np
"""
convert centernet-r50 pretrain model from torch to mindspore
"""
import argparse
from mindspore.train.serialization import save_checkpoint
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype


def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    print(type(par_dict))
    pt_params = {}
    for name, v in par_dict.items():
        pt_params[name] = v
        print(name, v)
    return pt_params


torch2ms = {}
torch_type2_ms_type = {
    "float32": mindspore.float32,
    "float16": mindspore.float16,
    "bfloat16": mindspore.bfloat16
}


def load_model_params2_dict(torch_name_file, ms_name_file):
    torch_names = []
    ms_names = []
    with open(torch_name_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            torch_names.append(line)
    with open(ms_name_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if 'backbone' in line:
                ms_names.append(line)
    if len(torch_names) != len(ms_names):
        raise ValueError("length of torch names and ms names should be equal")
    for i in range(len(torch_names)):
        torch2ms[torch_names[i]] = ms_names[i]


def convert(pth_file):
    pt_params_dict = pytorch_params(pth_file)
    weights = {}
    for name, value in pt_params_dict.items():
        if name not in torch2ms.keys():
            continue
        # key对应关系
        ms_name = torch2ms[name]
        # 先处理backbone
        weights[ms_name] = value.detach().numpy()

    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name], mstype.float32), name=name)
    param_list = []
    for key, value in parameter_dict.items():
        param_list.append({"name": key, "data": value})
    return param_list


def parse_args():
    parser = argparse.ArgumentParser(description="convert ckpt")
    parser.add_argument("--ckpt_file", required=True, help="ckpt file path")
    parser.add_argument("--torch_name_file", required=True, help="torch name file")
    parser.add_argument("--ms_name_file", required=True, help="mindspore name file")
    parser.add_argument("--output_file", required=True, help="output checkpoint file after convert")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    load_model_params2_dict(args.torch_name_file, args.ms_name_file)
    parameter_list = convert(args.ckpt_file)
    save_checkpoint(parameter_list, args.output_file)



