"""
Export CenterNet mindir model.
"""

import os
import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.centernet_det import CenterNetDetEvalV2, CenterNetDetEval
from src.model_utils.config import config, net_config, eval_config, export_config
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    export_config.ckpt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_config.ckpt_file)
    export_config.export_name = os.path.join(config.output_path, export_config.export_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export function'''
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=config.device_id)
    # net = CenterNetDetEvalV2(net_config, eval_config.K)
    net = CenterNetDetEval(net_config, eval_config.K)
    net.set_train(False)

    param_dict = load_checkpoint('/disk1/guoshipeng/models/research/cv/centernet_resnet50_v1/ckpt/centernet_last.ckpt')
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # img = Tensor(np.zeros([1, 3, 544, 544]), mindspore.float32)
    # img_shape = Tensor(np.zeros([2, ]), dtype=mindspore.float32)
    # border = Tensor(np.zeros([4, ]), dtype=mindspore.float32)
    # scale_factor = Tensor(np.zeros([4, ]), dtype=mindspore.float32)
    img = Tensor(np.zeros([1, 3, 512, 512]), dtype=mindspore.float32)
    # export(net, img, img_shape, border, scale_factor,
    #        file_name=export_config.export_name, file_format=export_config.export_format)
    export(net, img, file_name='/disk1/guoshipeng/models/research/cv/centernet_resnet50_v1/eval.mindir', file_format=export_config.export_format)


if __name__ == '__main__':
    run_export()
