import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter


class BatchNorm2dMasked(nn.Cell):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.9,
        affine=True,
        gamma_init="ones",
        beta_init="zeros",
        moving_mean_init="zeros",
        moving_var_init="ones",
        use_batch_statistics=None,
        data_format="NCHW",
    ):
        """Initialize _BatchNorm."""
        super(BatchNorm2dMasked, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.moving_mean = Parameter(
            initializer(moving_mean_init, num_features),
            name="mean",
            requires_grad=False,
        )
        self.moving_variance = Parameter(
            initializer(moving_var_init, num_features),
            name="variance",
            requires_grad=False,
        )
        self.gamma = Parameter(
            initializer(gamma_init, num_features), name="gamma", requires_grad=affine
        )
        self.beta = Parameter(
            initializer(beta_init, num_features), name="beta", requires_grad=affine
        )
        self.shape = P.Shape()
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.reduce_sum = P.ReduceSum()
        self.momentum = momentum
        self.use_batch_statistics = use_batch_statistics

    #    def construct(self, x, mask):
    #        mask = mask.reshape(-1, 1, 1, 1)
    #        mask = P.Cast()(mask, mstype.float32)
    #        #_shape_check(x.shape, mask.shape)
    #        point_mul = x * mask                                       # (n, c, h, w)
    #        all_sum = self.reduce_sum(point_mul, (0, 2, 3))            # (c)
    #        num = self.reduce_sum(mask) * x.shape[2] * x.shape[3]      #  1
    #        cur_mean = all_sum / num                                   # (c)
    #        cur_var = self.reduce_sum(self.square(x - cur_mean.reshape(1, -1, 1, 1)) * mask, (0, 2, 3)) / num  # (c)
    #        if self.use_batch_statistics is None:
    #            if self.training:
    #                self.moving_mean = self.moving_mean * self.momentum + cur_mean * (1 - self.momentum)
    #                self.moving_variance = self.moving_variance * self.momentum + cur_var * (1 - self.momentum)
    #            else:
    #                cur_mean = self.moving_mean
    #                cur_var = self.moving_variance
    #
    #        if self.use_batch_statistics:
    #            self.moving_mean = self.moving_mean * self.momentum + cur_mean * (1 - self.momentum)
    #            self.moving_variance = self.moving_variance * self.momentum + cur_var * (1 - self.momentum)
    #
    #
    #        # y = ((x - mean) / sqrt) * gamma + beta
    #        x = (x - self.reshape(cur_mean, (1, -1, 1, 1))) / self.sqrt(
    #            self.reshape(cur_var, (1, -1, 1, 1)) + self.eps)
    #        output = x * self.reshape(self.gamma, (1, -1, 1, 1)) + self.reshape(self.beta, (1, -1, 1, 1))
    #        output = output * mask
    #        return output

    def construct(self, x, mask):
        # x: [n, l, c], n:batch_size, l:length, c:channel
        mask = mask.reshape(-1, 1, 1)
        mask = P.Cast()(mask, mstype.float32)
        # _shape_check(x.shape, mask.shape)
        point_mul = x * mask  # (n, l, c)
        all_sum = self.reduce_sum(point_mul, (0, 1))  # (c)
        num = self.reduce_sum(mask) * x.shape[1]  # 1
        cur_mean = all_sum / num  # (c)
        cur_var = (
            self.reduce_sum(self.square(x - cur_mean.reshape(1, 1, -1)) * mask, (0, 1))
            / num
        )  # (c)
        if self.use_batch_statistics is None:
            if self.training:
                self.moving_mean = self.moving_mean * self.momentum + cur_mean * (
                    1 - self.momentum
                )
                self.moving_variance = (
                    self.moving_variance * self.momentum + cur_var * (1 - self.momentum)
                )
            else:
                cur_mean = self.moving_mean
                cur_var = self.moving_variance

        if self.use_batch_statistics:
            self.moving_mean = self.moving_mean * self.momentum + cur_mean * (
                1 - self.momentum
            )
            self.moving_variance = self.moving_variance * self.momentum + cur_var * (
                1 - self.momentum
            )

        # y = ((x - mean) / sqrt) * gamma + beta
        x = (x - self.reshape(cur_mean, (1, 1, -1))) / self.sqrt(
            self.reshape(cur_var, (1, 1, -1)) + self.eps
        )
        output = x * self.reshape(self.gamma, (1, 1, -1)) + self.reshape(
            self.beta, (1, 1, -1)
        )
        output = output * mask
        return output
