"""
CenterNet for training and evaluation
"""
import time

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore import amp
from mindspore.ops import operations as P
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.common.initializer import Constant
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from src.utils import Sigmoid, GradScale
from src.utils import FocalLoss, RegLoss
from src.decode import DetectionDecode
from src.resnet import BasicBlock, ResNet, ModulatedDeformConv2d
from .model_utils.config import dataset_config as data_cfg
import mindspore as ms
from mindspore.ops import Print
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.initializer import Normal
BN_MOMENTUM = 0.9

GRADIENT_CLIP_TYPE = 2
GRADIENT_CLIP_VALUE = 35

clip_grad = C.MultitypeFuncGraph("clip_grad")


def _generate_feature(cin, cout, kernel_size, head_name, head_conv=0):
    """
    Generate ResNet feature extraction function of each target head
    """
    fc = None
    if 'hm' in head_name:
        conv2d = nn.Conv2d(head_conv, cout, kernel_size=kernel_size, has_bias=True, bias_init=Constant(-2.19))
        fc = nn.SequentialCell([nn.Conv2d(cin, head_conv, kernel_size=3, has_bias=True), nn.ReLU(), conv2d])
    else:
        layers = []
        layers.append(nn.Conv2d(cin, head_conv, kernel_size=3, has_bias=True, weight_init=Normal(0.001),
                                bias_init=Constant(0)))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(head_conv, cout, kernel_size=kernel_size, has_bias=True, weight_init=Normal(0.001),
                                bias_init=Constant(0)))
        fc = nn.SequentialCell(layers)
    return fc


def resnet18(pretrained=True, backbone_ckpt=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained and backbone_ckpt is not None:
        ms.load_checkpoint(backbone_ckpt, model)
    return model


class GatherDetectionFeatureCell(nn.Cell):
    """
    Gather ResNet features of multi-pose estimation.

    Args:
        net_config: The config info of CenterNet network.

    Returns:
        Tuple of Tensors, the target head of multi-person pose.
    """

    def __init__(self, net_config, is_train=False):
        super(GatherDetectionFeatureCell, self).__init__()
        heads = {'hm': data_cfg.num_classes, 'wh': 2}
        if net_config.reg_offset:
            heads.update({'reg': 2})
        head_conv = net_config.head_conv
        self.backbone = resnet18(is_train, net_config.load_backbone_path)

        # neck
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

        self.hm_fn = _generate_feature(cin=64, cout=heads['hm'], kernel_size=1,
                                       head_name='hm', head_conv=head_conv)
        self.wh_fn = _generate_feature(cin=64, cout=heads['wh'], kernel_size=1,
                                       head_name='wh', head_conv=head_conv)
        if net_config.reg_offset:
            self.reg_fn = _generate_feature(cin=64, cout=heads['reg'], kernel_size=1,
                                            head_name='reg', head_conv=head_conv)
        self.reg_offset = net_config.reg_offset
        self.not_enable_mse_loss = not net_config.mse_loss
        self.Sigmoid = Sigmoid()

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Make deconvolution network of ResNet.

        Args:
            num_layer(int): Layer number.
            num_filters (list): Convolution dimension.
            num_kernels (list): The size of convolution kernel .

        Returns:
            SequentialCell, the output layer.
        """
        layers = []
        cin = 512
        for i in range(num_layers):
            kernel = num_kernels[i]
            cout = num_filters[i]
            conv_module = ModulatedDeformConv2d(cin, cout, kernel_size=3, padding=1, has_bias=False)
            layers.append(conv_module)
            layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU())
            up = nn.Conv2dTranspose(in_channels=cout, out_channels=cout,
                                    kernel_size=kernel, stride=2,
                                    pad_mode='pad', padding=1)

            layers.append(up)
            layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU())
            cin = cout

        return nn.SequentialCell(*layers)

    def construct(self, image):
        """Defines the computation performed."""
        output = self.backbone(image)
        output = self.deconv_layers(output[-1])
        feature = ()
        out = {}
        out['hm'] = self.hm_fn(output)
        out['hm'] = self.Sigmoid(out['hm'])
        out['wh'] = self.wh_fn(output)
        if self.reg_offset:
            out['reg'] = self.reg_fn(output)
        feature += (out,)
        return feature


class CenterNetLossCell(nn.Cell):
    """
    Provide object detection network losses.

    Args:
        net_config: The config info of CenterNet network.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, net_config):
        super(CenterNetLossCell, self).__init__()
        self.network = GatherDetectionFeatureCell(net_config, is_train=True)
        self.num_stacks = net_config.num_stacks
        self.reduce_sum = ops.ReduceSum()
        self.Sigmoid = Sigmoid()
        self.FocalLoss = FocalLoss()
        self.crit = nn.MSELoss() if net_config.mse_loss else self.FocalLoss
        self.crit_reg = RegLoss(net_config.reg_loss)
        self.crit_wh = RegLoss(net_config.reg_loss)
        self.num_stacks = net_config.num_stacks
        self.wh_weight = net_config.wh_weight
        self.hm_weight = net_config.hm_weight
        self.off_weight = net_config.off_weight
        self.reg_offset = net_config.reg_offset
        self.not_enable_mse_loss = not net_config.mse_loss
        self.print = ops.Print()

    def construct(self, image, hm, reg_mask, ind, wh, reg):
        """Defines the computation performed."""
        # breakpoint()
        hm_loss, wh_loss, off_loss = 0, 0, 0
        feature = self.network(image)

        for s in range(self.num_stacks):
            output = feature[s]
            # if self.not_enable_mse_loss:
            #     output_hm = self.Sigmoid(output['hm'])
            # else:
            # TODO 对比mmdet, sigmoid放到construct中
            output_hm = output['hm']
            hm_loss += self.crit(output_hm, hm) / self.num_stacks

            output_wh = output['wh']
            wh_loss += self.crit_reg(output_wh, reg_mask, ind, wh) / self.num_stacks

            if self.reg_offset and self.off_weight > 0:
                output_reg = output['reg']
                off_loss += self.crit_reg(output_reg, reg_mask, ind, reg) / self.num_stacks
        total_loss = (self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss)
        self.print(hm_loss)
        self.print(wh_loss)
        self.print(off_loss)
        return total_loss


class ImagePreProcess(nn.Cell):
    """
    Preprocess of image on device inplace of on host to improve performance.

    Args: None

    Returns:
        Tensor, normlized images and the format were converted to be NCHW
    """

    def __init__(self):
        super(ImagePreProcess, self).__init__()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 3, 1, 2)
        self.mean = Tensor(data_cfg.mean.reshape((1, 1, 1, 3)))
        self.std = Tensor(data_cfg.std.reshape((1, 1, 1, 3)))
        self.cast = ops.Cast()

    def construct(self, image):
        image = self.cast(image, mstype.float32)
        image = (image - self.mean) / self.std
        image = self.transpose(image, self.perm_list)
        return image


class CenterNetWithoutLossScaleCell(nn.Cell):
    """
    Encapsulation class of centernet training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.

    Returns:
        Tuple of Tensors, the loss, overflow flag and scaling sens of the network.
    """

    def __init__(self, network, optimizer):
        super(CenterNetWithoutLossScaleCell, self).__init__(auto_prefix=False)
        self.image = ImagePreProcess()
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)

    @ops.add_flags(has_effect=True)
    def construct(self, image, hm, reg_mask, ind, wh, reg):
        """Defines the computation performed."""
        image = self.image(image)
        weights = self.weights
        loss = self.network(image, hm, reg_mask, ind, wh, reg)
        grads = self.grad(self.network, weights)(image, hm, reg_mask, ind, wh, reg)
        succ = self.optimizer(grads)
        ret = loss
        return ops.depend(ret, succ)


class CenterNetWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of centernet training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (number): Static loss scale. Default: 1.

    Returns:
        Tuple of Tensors, the loss, overflow flag and scaling sens of the network.
    """

    def __init__(self, network, optimizer, sens=1):
        super(CenterNetWithLossScaleCell, self).__init__(auto_prefix=False)
        # self.image = ImagePreProcess()
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.learning_rate = self.optimizer.learning_rate
        self.global_step = self.optimizer.global_step
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.allreduce = ops.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = ops.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = ops.Cast()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.grad_scale = GradScale()
        self.loss_scale = sens

    @ops.add_flags(has_effect=True)
    def construct(self, image, hm, reg_mask, ind, wh, reg):
        """Defines the computation performed."""
        lr = self.learning_rate(self.global_step)
        # image = self.image(image)
        weights = self.weights
        loss = self.network(image, hm, reg_mask, ind, wh, reg)
        scaling_sens = self.cast(self.loss_scale, mstype.float32) * 2.0 / 2.0
        # alloc status and clear should be right before gradoperation
        grads = self.grad(self.network, weights)(image, hm, reg_mask, ind, wh, reg, scaling_sens)
        grads = self.grad_reducer(grads)
        grads = self.grad_scale(scaling_sens * self.degree, grads)
        overflow = ops.logical_not(amp.all_finite(grads))
        if self.reducer_flag:
            overflow = self.allreduce(overflow.to(mstype.float32)) >= self.base
        if not overflow:
            self.optimizer(grads)
        # ret = (loss, overflow, scaling_sens, lr)
        ret = (loss, overflow, scaling_sens, lr)
        return ret


class CenterNetDetEval(nn.Cell):
    """
    Encapsulation class of centernet testing.

    Args:
        net_config: The config info of CenterNet network.
        K(number): Max number of output objects. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: False.

    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """

    def __init__(self, net_config, K=100, enable_nms_fp16=False):
        super(CenterNetDetEval, self).__init__()
        self.network = GatherDetectionFeatureCell(net_config)
        self.decode = DetectionDecode(net_config, 100, enable_nms_fp16)
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()

    def construct(self, image):
        """Calculate prediction scores"""
        output = self.network(image)
        features = output[-1]
        detections = self.decode(features)
        return detections

