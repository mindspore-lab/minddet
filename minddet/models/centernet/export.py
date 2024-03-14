"""
Export CenterNet mindir model.
"""
import os

import mindspore
import numpy as np
from mindspore import Tensor, context
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from src.centernet_det import CenterNetDetEval
from src.model_utils.config import config, eval_config, export_config, net_config
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    """modelarts pre process function."""
    export_config.ckpt_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), export_config.ckpt_file
    )
    export_config.export_name = os.path.join(
        config.output_path, export_config.export_name
    )


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """export function"""
    context.set_context(
        mode=context.GRAPH_MODE, device_target="Ascend", device_id=config.device_id
    )
    net = CenterNetDetEval(net_config, eval_config.K)
    net.set_train(False)

    param_dict = load_checkpoint(export_config.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    img = Tensor(
        np.zeros(
            [
                1,
                3,
                export_config.dataset_config.input_res_test[0],
                export_config.dataset_config.input_res_test[1],
            ]
        ),
        dtype=mindspore.float32,
    )
    export(
        net,
        img,
        file_name=export_config.export_name,
        file_format=export_config.export_format,
    )


if __name__ == "__main__":
    run_export()
