import mindspore
import torch

"""
convert centernet-r50 pretrain model from torch to mindspore
"""
import argparse

from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import save_checkpoint


def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location="cpu")["state_dict"]
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        # print(name)
        pt_params[name] = parameter.numpy()
    return pt_params


def ms_params(network):
    with open("ms_name.txt", "w") as f:
        for k, v in network.parameters_dict().items():
            print(k, v.shape)
            f.write(k + "\r\n")


torch2ms = {}
torch_type2_ms_type = {
    "float32": mindspore.float32,
    "float16": mindspore.float16,
    "bfloat16": mindspore.bfloat16,
}


def load_model_params2_dict(torch_name_file, ms_name_file):
    torch_names = []
    ms_names = []
    with open(torch_name_file, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            torch_names.append(line)
    with open(ms_name_file, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            ms_names.append(line)
    if len(torch_names) != len(ms_names):
        raise ValueError("length of torch names and ms names should be equal")
    for i in range(len(torch_names)):
        torch2ms[torch_names[i]] = ms_names[i]


def convert(pth_file):
    pt_params_dict = pytorch_params(pth_file)
    weights = {}
    for name, value in pt_params_dict.items():
        print(name)
        if name not in torch2ms.keys():
            continue
        # key对应关系
        ms_name = torch2ms[name]
        # 先处理backbone
        print("-----before", name, ms_name)
        if "moving_mean" in ms_name:
            print(1)
            ms_name = ms_name.replace("moving_mean", "gamma")
        elif "moving_variance" in ms_name:
            print(2)
            ms_name = ms_name.replace("moving_variance", "beta")
        elif "gamma" in ms_name:
            print(3)
            ms_name = ms_name.replace("gamma", "moving_mean")
        elif "beta" in ms_name:
            print(4)
            ms_name = ms_name.replace("beta", "moving_variance")
        print("-----after", name, ms_name)

        weights[ms_name] = value

    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(
            Tensor(weights[name], torch_type2_ms_type[str(weights[name].dtype)]),
            name=name,
        )
    param_list = []
    for key, value in parameter_dict.items():
        param_list.append({"name": key, "data": value})
    return param_list


def parse_args():
    parser = argparse.ArgumentParser(description="convert ckpt")
    parser.add_argument("--ckpt_file", required=True, help="ckpt file path")
    parser.add_argument("--torch_name_file", required=True, help="torch name file")
    parser.add_argument("--ms_name_file", required=True, help="mindspore name file")
    parser.add_argument(
        "--output_file", required=True, help="output checkpoint file after convert"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    load_model_params2_dict(args.torch_name_file, args.ms_name_file)
    parameter_list = convert(args.ckpt_file)
    save_checkpoint(parameter_list, args.output_file)
