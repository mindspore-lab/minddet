"""
Functional Cells to be used.
"""

import math
import time
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from mindspore.train.callback import Callback

reciprocal = ops.Reciprocal()
grad_scale = ops.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class GradScale(nn.Cell):
    """
    Gradients scale

    Args: None

    Returns:
        Tuple of Tensors, gradients after rescale.
    """

    def __init__(self):
        super(GradScale, self).__init__()
        self.hyper_map = ops.HyperMap()

    def construct(self, scale, grads):
        grads = self.hyper_map(ops.partial(grad_scale, scale), grads)
        return grads


class GatherFeature(nn.Cell):
    """
    Gather feature at specified position

    Args:
        enable_cpu_gather (bool): Use cpu operator GatherD to gather feature or not, adaption for CPU. Default: False.

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self, enable_cpu_gather=False):
        super(GatherFeature, self).__init__()
        self.tile = ops.Tile()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.enable_cpu_gather = enable_cpu_gather
        self.start = Tensor(0, mstype.int32)
        self.step = Tensor(1, mstype.int32)
        if self.enable_cpu_gather:
            self.gather_nd = ops.GatherD()
            self.expand_dims = ops.ExpandDims()
        else:
            self.gather_nd = ops.GatherNd()
        self.cast = ops.Cast()
        self.print = ops.Print()

    def construct(self, feat, ind):
        """gather by specified index"""
        # breakpoint()
        if self.enable_cpu_gather:
            _, _, c = self.shape(feat)
            # (b, N, c)
            index = self.expand_dims(ind, -1)
            index = self.tile(index, (1, 1, c))
            feat = self.gather_nd(feat, 1, index)
        else:
            # (b, N)->(b*N, 1)
            b, N = self.shape(ind)
            ind = self.reshape(ind, (-1, 1))
            ind_b = ops.range(self.start, Tensor(b, mstype.int32), self.step).astype(mstype.int32)
            ind_b = self.reshape(ind_b, (-1, 1))
            ind_b = self.tile(ind_b, (1, N))
            ind_b = self.reshape(ind_b, (-1, 1))
            ind_b = self.cast(ind_b, mstype.int32)
            ind = self.cast(ind, mstype.int32)
            index = self.concat((ind_b, ind))
            index = self.reshape(index, (b, N, -1))
            # (b, N, c)
            feat = self.gather_nd(feat, index)
        return feat


class TransposeGatherFeature(nn.Cell):
    """
    Transpose and gather feature at specified position

    Args: None

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self):
        super(TransposeGatherFeature, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 3, 1)
        self.gather_feat = GatherFeature()

    def construct(self, feat, ind):
        # (b, c, h, w)->(b, h*w, c)
        feat = self.transpose(feat, self.perm_list)
        b, _, _, c = self.shape(feat)
        feat = self.reshape(feat, (b, -1, c))
        # (b, N, c)
        feat = self.gather_feat(feat, ind)
        return feat


class Sigmoid(nn.Cell):
    """
    Sigmoid and then Clip by value

    Args: None

    Returns:
        Tensor, feature after sigmoid and clip.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.sigmoid = nn.Sigmoid()
        self.clip_by_value = ops.clip_by_value

    def construct(self, x, min_value=1e-4, max_value=1 - 1e-4):
        x = self.sigmoid(x)
        dt = self.dtype(x)
        x = self.clip_by_value(x, self.cast(ops.tuple_to_array((min_value,)), dt),
                               self.cast(ops.tuple_to_array((max_value,)), dt))
        return x


class FocalLoss(nn.Cell):
    """
    Warpper for focal loss.

    Args:
        alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
        beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples. Default: 4.

    Returns:
        Tensor, focal loss.
    """

    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.select = ops.Select()
        self.equal = ops.Equal()
        self.less = ops.Less()
        self.cast = ops.Cast()
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, out, target):
        """focal loss"""
        pos_inds = self.cast(self.equal(target, 1.0), mstype.float32)
        neg_inds = self.cast(self.less(target, 1.0), mstype.float32)
        neg_weights = self.pow(1 - target, self.beta)

        pos_loss = self.log(out) * self.pow(1 - out, self.alpha) * pos_inds
        neg_loss = self.log(1 - out) * self.pow(out, self.alpha) * neg_weights * neg_inds

        num_pos = self.reduce_sum(pos_inds, ())
        num_pos = self.select(self.equal(num_pos, 0.0),
                              self.fill(self.dtype(num_pos), self.shape(num_pos), 1.0), num_pos)
        pos_loss = self.reduce_sum(pos_loss, ())
        neg_loss = self.reduce_sum(neg_loss, ())
        loss = - (pos_loss + neg_loss) / num_pos
        return loss


class RegLoss(nn.Cell):
    """
    Warpper for regression loss.

    Args:
        mode(str): L1 or Smoothed L1 loss. Default: "l1"

    Returns:
        Tensor, regression loss.
    """

    def __init__(self, mode='l1'):
        super(RegLoss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.gather_feature = TransposeGatherFeature()
        if mode == 'l1':
            self.loss = nn.L1Loss(reduction='sum')
        elif mode == 'sl1':
            self.loss = nn.SmoothL1Loss()
        else:
            self.loss = None

    def construct(self, output, mask, ind, target):
        pred = self.gather_feature(output, ind)
        mask = self.cast(mask, mstype.float32)

        num = self.reduce_sum(mask, ()) * 2
        mask = self.expand_dims(mask, 2)
        target = target * mask
        pred = pred * mask
        regr_loss = self.loss(pred, target)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss


class TimeMonitor(Callback):
    """
    Monitor the time in train or eval process.
    """

    def __init__(self, data_size=None, data_time=False):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()
        self.data_time = data_time
        self.data_time_sum = 0.0
        self.data_time_start = 0.0
        self.data_sink = lambda c: c.original_args()["dataset_sink_mode"]
        # Validator.check_bool(data_time, "data_time")
        self.step_time = 0

    def on_train_step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        if self.data_time and not self.data_sink(run_context):
            interval = time.time() - self.data_time_start
            self.data_time_sum = self.data_time_sum + interval
        self.step_time = time.time()

    def on_train_step_end(self, run_context):
        """
        Record time at the end of step.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        if self.data_time and not self.data_sink(run_context):
            self.data_time_start = time.time()
        print(f"step consumes: {time.time() - self.step_time}", flush=True)

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        self.epoch_time = time.time()
        if self.data_time and not self.data_sink(run_context):
            self.data_time_sum = 0.0
            self.data_time_start = time.time()

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.

        Args:
           run_context (RunContext): Context of the process running. For more details,
                   please refer to :class:`mindspore.train.RunContext`.
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        mode = cb_params.get("mode", "")
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num
        # Validator.check_positive_int(step_size)

        step_seconds = epoch_seconds / step_size

        train_log = "{} epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(
            mode.title(), epoch_seconds, step_seconds)

        if self.data_time and not self.data_sink(run_context):
            data_step_seconds = self.data_time_sum * 1000 / step_size
            data_log = " (data time: {:5.3f} ms)".format(data_step_seconds)
            train_log += data_log
        elif self.data_time and self.data_sink(run_context):
            # send info viewer to query epoch message of cur_epoch_num
            send_info = cb_params["dataset_helper"].get_send_info(run_context)
            epoch = cb_params["cur_epoch_num"]
            epoch_send_info = send_info.epoch(epoch)
            # show average time of fetching data time
            fetch_data_time = epoch_send_info["fetch_data_time"]
            data_log = " (data time: {:5.3f} ms)".format(fetch_data_time)
            train_log += data_log

        print(train_log, flush=True)


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.

    Args:
        dataset_size (int): Dataset size. Default: -1.
        enable_static_time (bool): enable static time cost, adaption for CPU. Default: False.
    """

    def __init__(self, dataset_size=-1, enable_static_time=False):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self._enable_static_time = enable_static_time

    def on_train_step_begin(self, run_context):
        self._begin_time = time.time()

    def on_train_step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            if self._enable_static_time:
                cur_time = time.time()
                time_per_step = cur_time - self._begin_time
                print("epoch: {}, current epoch percent: {}, step: {}, time per step: {} s, outputs are {}"
                      .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, "%.3f" % time_per_step,
                              str(cb_params.net_outputs)), flush=True)
            else:
                print(
                    "epoch: {} | current epoch percent: {} | step: {} | loss {} | overflow {} | scaling_sens {} | lr {}"
                        .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num,
                                str(cb_params.net_outputs[0].asnumpy()),
                                str(cb_params.net_outputs[1].asnumpy()),
                                str(cb_params.net_outputs[2].asnumpy()),
                                str(cb_params.net_outputs[3].asnumpy())), flush=True)
        else:
            print("epoch: {} | step: {} | loss {} | overflow {} | scaling_sens {} | lr {}"
                  "".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                            str(cb_params.net_outputs[0]),
                            str(cb_params.net_outputs[1]),
                            str(cb_params.net_outputs[2]),
                            str(cb_params.net_outputs[3])), flush=True)


class CenterNetPolynomialDecayLR(LearningRateSchedule):
    """
    Warmup and polynomial decay learning rate for CenterNet network.

    Args:
        learning_rate(float): Initial learning rate.
        end_learning_rate(float): Final learning rate after decay.
        warmup_steps(int): Warmup steps.
        decay_steps(int): Decay steps.
        power(int): Learning rate decay factor.

    Returns:
        Tensor, learning rate in time.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(CenterNetPolynomialDecayLR, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class CenterNetMultiEpochsDecayLR(LearningRateSchedule):
    """
    Warmup and multi-steps decay learning rate for CenterNet network.

    Args:
        learning_rate(float): Initial learning rate.
        warmup_steps(int): Warmup steps.
        multi_steps(list int): The steps corresponding to decay learning rate.
        steps_per_epoch(int): How many steps for each epoch.
        factor(int): Learning rate decay factor. Default: 10.

    Returns:
        Tensor, learning rate in time.
    """

    def __init__(self, learning_rate, warmup_steps, multi_epochs, steps_per_epoch, factor=10):
        super(CenterNetMultiEpochsDecayLR, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = MultiEpochsDecayLR(learning_rate, multi_epochs, steps_per_epoch, factor)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        # if self.warmup_flag:
        #     is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
        #     warmup_lr = self.warmup_lr(global_step)
        #     lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        # else:
        # lr = decay_lr
        # return lr
        return decay_lr


class MultiEpochsDecayLR(LearningRateSchedule):
    """
    Calculate learning rate base on multi epochs decay function.

    Args:
        learning_rate(float): Initial learning rate.
        multi_steps(list int): The steps corresponding to decay learning rate.
        steps_per_epoch(int): How many steps for each epoch.
        factor(int): Learning rate decay factor. Default: 10.

    Returns:
        Tensor, learning rate.
    """

    def __init__(self, learning_rate, multi_epochs, steps_per_epoch, factor=10):
        super(MultiEpochsDecayLR, self).__init__()
        if not isinstance(multi_epochs, (list, tuple)):
            raise TypeError("multi_epochs must be list or tuple.")
        self.multi_epochs = Tensor(np.array(multi_epochs, dtype=np.float32) * steps_per_epoch)
        self.num = len(multi_epochs)
        self.start_learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.factor = factor
        self.pow = ops.Pow()
        self.cast = ops.Cast()
        self.less_equal = ops.LessEqual()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, global_step):
        cur_step = self.cast(global_step, mstype.float32)
        multi_epochs = self.cast(self.multi_epochs, mstype.float32)
        epochs = self.cast(self.less_equal(multi_epochs, cur_step), mstype.float32)
        lr = self.start_learning_rate / self.pow(self.factor, self.reduce_sum(epochs, ()))
        return lr


class MultiStepWithLinearLR(LearningRateSchedule):
    """
    Linear with Warm Up Learning Rate.

    Args:
        learning_rate (`float`):
            Initial value of learning rate.
        warmup_steps (`int`):
            The number of warm up steps.
        total_steps (`int`):
            The number of total steps.
        warmup_lr_init (`float`, *optional*, defaults to 0.):
            Initial learning rate in warm up steps.

    Returns:
        Class, LinearWithWarmUpLR
    """

    def __init__(self, learning_rate: float, linear_steps: int,
                 start_linear_lr: float = 0.0, multi_epochs=None,
                 decay_factor=10, steps_per_epoch: int = 0):
        super(MultiStepWithLinearLR, self).__init__()
        self.start_lr = learning_rate
        # self.linear_steps = Tensor(linear_steps, mstype.float32)
        self.has_linear = linear_steps > 0
        self.linear_steps = Tensor(linear_steps, mstype.float32)
        self.greater = ops.Greater()
        self.max = ops.Maximum()
        self.steps_per_epoch = Tensor(steps_per_epoch, mstype.float32)
        self.start_linear_lr = Tensor(start_linear_lr, mstype.float32)
        self.decay_lr = MultiEpochsDecayLR(learning_rate, multi_epochs, steps_per_epoch, decay_factor)
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.multi_epochs = Tensor(np.array(multi_epochs, dtype=np.float32) * steps_per_epoch)
        self.factor = decay_factor
        self.pow = ops.Pow()
        self.cast = ops.Cast()
        self.less_equal = ops.LessEqual()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        # 先linear 再multi epoch decay
        decay_lr = self.decay_lr(global_step)
        # global_step = self.cast(global_step, mstype.float32)
        if self.has_linear:
            is_linear = self.cast(self.greater(self.linear_steps, global_step), mstype.float32)
            percent = self.max(self.zero_constant, (self.linear_steps - global_step) / self.linear_steps)
            linear_lr = (self.one - percent) * self.start_lr + self.start_linear_lr
            learning_rate = (self.one - is_linear) * decay_lr + is_linear * linear_lr
        else:
            learning_rate = decay_lr
        return learning_rate


class LinearWithWarmUpLR(LearningRateSchedule):
    """
    Linear with Warm Up Learning Rate.

    Args:
        learning_rate (`float`):
            Initial value of learning rate.
        warmup_steps (`int`):
            The number of warm up steps.
        total_steps (`int`):
            The number of total steps.
        warmup_lr_init (`float`, *optional*, defaults to 0.):
            Initial learning rate in warm up steps.

    Returns:
        Class, LinearWithWarmUpLR
    """

    def __init__(self, learning_rate: float, warmup_steps: int, total_steps: int,
                 warmup_lr_init: float = 0.):
        super(LinearWithWarmUpLR, self).__init__()
        linear_steps = max(1, total_steps - warmup_steps)
        warmup_steps = max(1, warmup_steps)
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.linear_steps = Tensor(linear_steps, mstype.float32)
        self.greater = ops.Greater()
        self.max = ops.Maximum()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = ops.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            percent = self.max(self.zero_constant, (self.total_steps - global_step) / self.linear_steps)
            learning_rate = self.learning_rate * percent
        return learning_rate
