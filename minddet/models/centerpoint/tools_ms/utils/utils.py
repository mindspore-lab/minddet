from __future__ import absolute_import

import time

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import Callback


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
        cb_params = run_context.original_args()
        if self.data_time and not self.data_sink(run_context):
            self.data_time_start = time.time()
        print(
            f"step consumes: {time.time() - self.step_time}, lr: {str(cb_params.net_outputs[1])}"
        )

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
            mode.title(), epoch_seconds, step_seconds
        )

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


class TimeMonitorEval(Callback):
    """
    Monitor the time in train or eval process.
    """

    def __init__(self, data_size=None, data_time=False):
        super(TimeMonitorEval, self).__init__()
        self.data_size = data_size
        self.epoch_time = 0
        self.step_time = 0
        self.step_time_total = 0

    def on_eval_begin(self, run_context):
        """
        Called before eval begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        # self.begin(run_context)
        print("before eval begin")

    def on_eval_epoch_begin(self, run_context):
        """
        Called before eval epoch begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        # self.epoch_begin(run_context)
        self.epoch_time = time.time()

    def on_eval_epoch_end(self, run_context):
        """
        Called after eval epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        # self.epoch_end(run_context)\
        self.epoch_time = time.time() - self.epoch_time
        print(f"Inference consumes {self.epoch_time}s")
        print(f"each step consumes averagely {self.step_time_total / self.data_size}")
        print(f"average step time by epoch time {self.epoch_time / self.data_size}")

    def on_eval_step_begin(self, run_context):
        """
        Called before each eval step begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        pass

    def on_eval_step_end(self, run_context):
        """
        Called after each eval step end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        pass

    def on_eval_end(self, run_context):
        """
        Called after eval end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        # self.end(run_context)
        print("after eval end")


class TrainOneStepCellWrapper(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, sens=1.0):
        if isinstance(sens, (int, float)):
            scale_sense = nn.FixedLossScaleUpdateCell(sens)
        super(TrainOneStepCellWrapper, self).__init__(network, optimizer, scale_sense)

        self.network = network
        # self.loss_cell = loss_cell
        self.optimizer = optimizer
        # self.weights = ParameterTuple(network.trainable_params())
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens

        self.network.set_grad()
        self.weights = optimizer.parameters

        self.learning_rate = self.optimizer.learning_rate
        self.global_step = self.optimizer.global_step
        self.print = ops.Print()

    def construct(
        self,
        voxels,
        coordinates,
        num_points,
        num_voxels,
        shape,
        hm_or_token=None,
        anno_box=None,
        ind=None,
        mask=None,
        cat=None,
    ):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(
            voxels,
            coordinates,
            num_points,
            num_voxels,
            shape,
            hm_or_token,
            anno_box,
            ind,
            mask,
            cat,
        )

        status, sens = self.start_overflow_check(loss, self.sens)  # todo added
        # loss = self.loss_cell(example, preds)
        # print(loss)
        # 为反向传播设定系数
        # start_time = time.time()
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        # sens = (sens, sens)
        grads = self.grad(self.network, weights)(
            voxels,
            coordinates,
            num_points,
            num_voxels,
            shape,
            hm_or_token,
            anno_box,
            ind,
            mask,
            cat,
            sens,
        )
        grads = ops.clip_by_norm(grads, 35, 2)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        grads = self.grad_reducer(grads)

        if not overflow:
            self.optimizer(grads)
        else:
            self.print("Loss overflows, skip update weights")
        lr = self.learning_rate(self.global_step)

        return loss, lr
