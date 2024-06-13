import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, ops
from mindspore.common import dtype as mstype

from ..registry import NECKS


@NECKS.register_module
class RPN(nn.Cell):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.99)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = self._upsample_strides[i - self._upsample_start_idx]
                if stride > 1:
                    deblock = nn.SequentialCell(
                        nn.Conv2dTranspose(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            has_bias=False,
                        ),
                        nn.BatchNorm2d(
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            eps=1e-3,
                            momentum=0.99,
                        ),
                        nn.ReLU(),
                    )
                else:
                    stride = int(np.round(1 / stride))
                    deblock = nn.SequentialCell(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            has_bias=False,
                            weight_init="xavier_uniform",
                        ),
                        nn.BatchNorm2d(
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            eps=1e-3,
                            momentum=0.99,
                        ),
                        # nn.SyncBatchNorm(self._num_upsample_filters[i - self._upsample_start_idx], eps=1e-3,
                        #                momentum=0.99),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.CellList(blocks)
        self.deblocks = nn.CellList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = [
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1))),
            nn.Conv2d(
                inplanes,
                planes,
                3,
                stride=stride,
                has_bias=False,
                pad_mode="pad",
                weight_init="xavier_uniform",
            ),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.99),
            # nn.SyncBatchNorm(planes, eps=1e-3, momentum=0.99),
            nn.ReLU(),
        ]

        for j in range(num_blocks):
            block.append(
                nn.Conv2d(
                    planes, planes, 3, has_bias=False, weight_init="xavier_uniform"
                )
            )
            block.append(
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.99)
                # nn.SyncBatchNorm(planes, eps=1e-3, momentum=0.99)
            )
            block.append(nn.ReLU())

        return nn.SequentialCell(block), planes

    def construct(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = ops.Concat(axis=1)(ups)

        return x


if __name__ == "__main__":
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    rpn = RPN([3, 5, 5], [2, 2, 2], [64, 128, 256], [0.5, 1, 2], [128, 128, 128], 64)
    batch_size = 4
    print(
        rpn(Tensor(np.zeros([batch_size, 64, 512, 512]), dtype=mstype.float32)).shape
        == (batch_size, 384, 128, 128)
    )  #
