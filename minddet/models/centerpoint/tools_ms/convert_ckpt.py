from mindspore import Tensor, save_checkpoint


def main():
    print("start convert")
    import torch

    path = "./locate_operator/step0.pth"
    data = torch.load(path, map_location=torch.device("cpu"))
    keys = sorted(data["state_dict"].keys())

    key_list = []
    key_list2 = []
    count = 0
    for item in sorted(keys):
        if "num_batches_tracked" in item or "global_step" in item:
            count += 1
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
    save_path = "./locate_operator/torch_weight_0.ckpt"
    save_checkpoint(key_list2, save_path)


if __name__ == "__main__":
    main()
