import math

import torch.nn as nn
from functools import partial

from models.utils import Swish, Sigmoid
from models.effnet import ArcMarginModel, SqueezeExcite

__all__ = ['ResNet']


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_block=nn.Conv2d,
                 norm_block=None, act_fn=None, **conv_params):
        super(Conv, self).__init__()
        padding = _get_padding(kernel_size, **conv_params)
        self.conv = conv_block(in_channels, out_channels, kernel_size, padding=padding, **conv_params)
        self.bn = None if norm_block is None else norm_block(out_channels)
        self.act_fn = None if act_fn is None else act_fn()

    def forward(self, input):
        output = self.conv(input)
        if self.bn is not None:
            output = self.bn(output)
        if self.act_fn is not None:
            output = self.act_fn(output)
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chs, out_chs, stride=1, downsample=None,
                 conv_block=Conv, act_fn=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.stride = stride

        self.block1 = conv_block(in_chs, out_chs, 3, stride=stride, bias=False, act_fn=act_fn)
        self.block2 = conv_block(out_chs, out_chs, 3, stride=1, bias=False)
        self.downsample = downsample
        self.act_fn = act_fn()

    def forward(self, x):
        residual = x

        out = self.block1(x)
        out = self.block2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fn(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chs, out_chs, stride=1, downsample=None,
                 conv_block=Conv, act_fn=nn.ReLU):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.block1 = conv_block(in_chs, out_chs, 1, bias=False, act_fn=act_fn)
        self.block2 = conv_block(out_chs, out_chs, 3, bias=False, stride=stride, act_fn=act_fn)
        self.block3 = conv_block(out_chs, out_chs * self.expansion, 1, bias=False)
        self.act_fn = act_fn()

    def forward(self, x):
        residual = x

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fn(out)
        return out


class InvertedResidual(nn.Module):
    """ Inverted residual block with SE"""
    expansion = 1

    def __init__(self, in_chs, out_chs, stride=1, downsample=None,
                 conv_block=Conv, act_fn=nn.ReLU):
        super(InvertedResidual, self).__init__()
        expansion_ratio = 4
        mid_chs = int(in_chs * expansion_ratio)
        self.has_residual = in_chs == out_chs and stride == 1
        self.act_fn = act_fn()

        # Point-wise expansion
        self.conv_pw = conv_block(in_chs, mid_chs, 1, bias=False, act_fn=act_fn)

        # Depth-wise convolution
        self.conv_dw = conv_block(mid_chs, mid_chs, 3, stride=stride, bias=False, groups=mid_chs, act_fn=act_fn)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_chs, reduce_chs=max(1, int(in_chs * 0.25)),
                                act_fn=self.act_fn, gate_fn=Sigmoid(True))

        # Point-wise linear projection
        self.conv_pwl = conv_block(mid_chs, out_chs, 1, bias=False)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)

        # Depth-wise convolution
        x = self.conv_dw(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)

        if self.has_residual:
            x += residual

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_chans=3, compression_type='none', reg_weight=1e-4,
                 norm_type='batch', norm_params=None, act_fn='relu', is_inplace=False):
        super(ResNet, self).__init__()

        self.choose_act_fn(act_fn, is_inplace)
        self.assembly_conv(compression_type, reg_weight, norm_type, norm_params)

        self.inplanes = 64
        self.num_features = 128 * block.expansion
        self.num_classes = num_classes
        margin = num_classes / (num_classes - 1) / 30
        margin *= 1 if num_classes <= self.num_features else self.num_features / num_classes
        scale = num_classes / (num_classes - 1) * math.log((num_classes - 1) * 0.9 / (1 - 0.9))

        self.layer0 = self.conv_block(in_chans, self.inplanes, 7, stride=2, bias=False, act_fn=self.act_fn)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 96, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 192, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier_arc_margin = ArcMarginModel(self.num_features, self.num_classes, margin=margin, scale=scale)
        self.classifier = nn.Linear(self.num_features, num_classes)

    def _make_layer(self, block, out_chs, num_blocks, stride=1):
        downsample = None
        inplanes = out_chs * block.expansion
        if stride != 1 or self.inplanes != inplanes:
            downsample = self.conv_block(self.inplanes, inplanes, 1, stride=stride, bias=False)

        block = partial(block, out_chs=out_chs, conv_block=self.conv_block, act_fn=self.act_fn)

        layers = [block(self.inplanes, stride=stride, downsample=downsample)]
        layers += [block(inplanes) for i in range(num_blocks - 1)]
        self.inplanes = inplanes

        return nn.Sequential(*layers)

    def assembly_conv(self, compression_type, reg_weight, norm_type, norm_params):
        if compression_type == 'vard':
            # conv_block = partial(Conv2dVARD, reg_weight=reg_weight)
            conv_block = nn.Conv2d
        elif compression_type == 'none':
            conv_block = nn.Conv2d
        else:
            raise ValueError('Incorrect compression_type value.')

        if norm_params is None:
            norm_params = {}

        if norm_type is None:
            norm_block = None
        elif norm_type == 'batch':
            norm_block = partial(nn.BatchNorm2d, **norm_params)
        elif norm_type == 'group':
            norm_block = partial(nn.GroupNorm, **norm_params)
        elif norm_type == 'instance':
            norm_block = partial(nn.InstanceNorm2d, **norm_params)
        elif norm_type == 'layer':
            norm_block = partial(nn.LayerNorm, **norm_params)
        else:
            raise ValueError(f'{norm_type} normalization is not supported')
        self.conv_block = partial(Conv, conv_block=conv_block, norm_block=norm_block)

    def choose_act_fn(self, act_fn, inplace=True):
        if act_fn is None:
            self.act_fn = None
        elif act_fn == 'relu':
            self.act_fn = partial(nn.ReLU, inplace=inplace)
        elif act_fn == 'swish':
            self.act_fn = partial(Swish, inplace=inplace)
        elif act_fn == 'sigmoid':
            self.act_fn = partial(Sigmoid, inplace=inplace)
        else:
            raise ValueError(f'{act_fn} is not supported')

    def forward_features(self, input):
        x = self.layer0(input)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

    def forward_two_heads(self, x, target=None):
        x = self.forward_features(x)
        x_arcface = self.classifier_arc_margin(x, target)
        x_alter = self.classifier(x)
        return x_arcface, x_alter


def resnet18(**kwargs):
    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = ResNet(InvertedResidual, [2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
