from functools import partial

import numpy as np


class LRSchedulerStep(object):
    def __init__(self, total_step, lr_phases, mom_phases):
        # self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < int(start * total_step)
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append(
                    (
                        int(start * total_step),
                        int(lr_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.lr_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append(
                    (
                        int(start * total_step),
                        int(mom_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.mom_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        # assert self.mom_phases[0][0] == 0
        if len(mom_phases) > 0:
            assert self.mom_phases[0][0] == 0

    def step(self, step):
        lrs, moms = [], []
        for start, end, func in self.lr_phases:
            if step >= start:
                # self.optimizer.lr = func((step - start) / (end - start))
                lrs.append(func((step - start) / (end - start)))
        # if len(lrs) > 0:
        #    self.optimizer.lr = lrs[-1]
        for start, end, func in self.mom_phases:
            if step >= start:
                moms.append(func((step - start) / (end - start)))
                # self.optimizer.mom = func((step - start) / (end - start))
        # if len(moms) > 0:
        #    self.optimizer.mom = moms[-1]
        return lrs[-1], moms[-1]

    def get_lr(self):
        lrs = []
        moms = []
        for i in range(self.total_step):
            lr, mom = self.step(i)
            lrs.append(lr)
            moms.append(mom)
        return np.array(lrs).astype(np.float32), np.array(moms).astype(np.float32)


def annealing_cos(start, end, pct):
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    def __init__(self, total_step, lr_max, div_factor, pct_start, moms=[0.95, 0.85]):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        low_lr = self.lr_max / self.div_factor
        lr_phases = (
            (0, partial(annealing_cos, low_lr, self.lr_max)),
            (self.pct_start, partial(annealing_cos, self.lr_max, low_lr / 1e4)),
        )
        mom_phases = (
            (0, partial(annealing_cos, *self.moms)),
            (self.pct_start, partial(annealing_cos, *self.moms[::-1])),
        )
        # fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        # super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)
        super().__init__(total_step, lr_phases, mom_phases)
