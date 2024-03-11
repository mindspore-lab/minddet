"""ResNet & ResNet-DCNv2"""

import math

import mindspore
from mindspore import ops, nn
from mindspore.common.initializer import Normal, HeNormal


# set initializer to constant for debugging.
def conv3x3(inplanes, outplanes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, pad_mode="pad",
                     padding=1, weight_init=HeNormal())


class ModulatedDeformConv2d(nn.Conv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.
    Args:
        in_channels (int): Same as nn.Conv2d. The channel number of the input tensor of the Conv2d layer.
        out_channels (int): Same as nn.Conv2d. The channel number of the output tensor of the Conv2d layer.
        kernel_size (int or tuple[int]): Same as nn.Conv2d. Specifies the height and width of the 2D convolution kernel.
        stride (int): Same as nn.Conv2d, while tuple is not supported. Default: 1.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        has_bias (bool: Same as nn.Conv2d. False.

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, has_bias=False):
        super(ModulatedDeformConv2d, self).__init__(in_channels,
                                                    out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    pad_mode="pad",
                                                    padding=padding,
                                                    has_bias=has_bias)
        self.deform_groups = 1
        self.de_stride = (1, 1, stride, stride)
        self.de_padding = (padding, padding, padding, padding)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad_mode="pad",
            padding=self.padding,
            dilation=self.dilation,
            has_bias=True,
            weight_init="Zero",
        )
        self.print = ops.Print()
        self.flatten = ops.Flatten()
        self.reshape = ops.Reshape()
        # self.concat = ops.Concat()
        self.transpose = ops.Transpose()

    def construct(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = ops.chunk(out, 3, axis=1)
        mask = ops.Sigmoid()(mask)

        # transform to ms data type
        batch, y, out_h, out_w = out.shape
        ms_off = ops.concat((o1, o2), axis=1)
        ms_off = self.flatten(ms_off)
        ms_off = self.reshape(ms_off, (batch, 1, 3, 3, 2, out_h, out_w))
        ms_off = self.transpose(ms_off, (0, 4, 1, 2, 3, 5, 6))
        offsets_y1, offsets_x1 = ops.chunk(ms_off, 2, axis=1)

        ms_mask = self.flatten(mask)
        ms_mask = self.reshape(ms_mask, (batch, 1, 3, 3, 1, out_h, out_w))
        ms_mask = self.transpose(ms_mask, (0, 4, 1, 2, 3, 5, 6))

        ms_offset = ops.concat((offsets_x1, offsets_y1, ms_mask), axis=1)
        ms_offset = self.reshape(ms_offset, (batch, 3 * 1 * 3 * 3, out_h, out_w))

        out = ops.deformable_conv2d(x, self.weight, ms_offset, self.kernel_size, self.de_stride, self.de_padding,
                                    bias=self.bias, deformable_groups=self.deform_groups, modulated=True)
        return out


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        if dcn:
            self.conv2 = ModulatedDeformConv2d(planes, planes, kernel_size=3, padding=1)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if dcn:
            self.conv2 = ModulatedDeformConv2d(planes, planes, kernel_size=3, padding=1, stride=stride)
        else:
            self.conv2 = conv3x3(planes, planes, stride=stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()

        self.downsample = downsample

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, layers, dcn=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.print = ops.Print()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad",
                               has_bias=False, weight_init=HeNormal())
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.SequentialCell([
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
            nn.MaxPool2d(kernel_size=3, stride=2)])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_init = Normal(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample的功能是调整residual使其和out保持相同尺寸，out的变化由plane和stride控制
            downsample = nn.SequentialCell(
                # set initializer to constant for debugging.
                nn.Conv2d(self.inplanes, planes * block.expansion, pad_mode="pad",
                          kernel_size=1, stride=stride, has_bias=False, weight_init=HeNormal()),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x.astype(mindspore.float16)).astype(mindspore.float32)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5
