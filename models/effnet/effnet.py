import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from models.utils import load_pretrained, swish, sigmoid, drop_connect


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, bn2d, dw_kernel_size=3, drop_connect_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = in_chs == out_chs
        self.act_fn = swish
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=1, bias=False,
                                 padding=_get_padding(dw_kernel_size), groups=in_chs)
        self.bn1 = bn2d(in_chs)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(in_chs, reduce_chs=max(1, int(in_chs * 0.25)),
                                act_fn=self.act_fn, gate_fn=sigmoid)

        self.conv_pw = nn.Conv2d(in_chs, out_chs, 1, bias=False)
        self.bn2 = bn2d(out_chs)

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class ArcMarginModel(nn.Module):
    def __init__(self, in_features, num_classes, margin, scale):
        super(ArcMarginModel, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.FloatTensor(num_classes))

        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, input, label=None):
        if not self.training or label is None:
            return F.linear(input, self.weight)

        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + margin)
        phi = torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = torch.where(one_hot == 1, phi, cosine)
        output *= self.scale
        return output


class InvertedResidual(nn.Module):
    """ Inverted residual block with SE"""

    def __init__(self, in_chs, out_chs, bn2d, dw_kernel_size=3,
                 stride=1, drop_connect_rate=0.):
        super(InvertedResidual, self).__init__()
        expansion_ratio = 6
        mid_chs = int(in_chs * expansion_ratio)
        self.has_residual = in_chs == out_chs and stride == 1
        self.act_fn = swish
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = nn.Conv2d(in_chs, mid_chs, 1, bias=False)
        self.bn1 = bn2d(mid_chs)

        # Depth-wise convolution
        self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, bias=False,
                                 padding=_get_padding(dw_kernel_size), groups=mid_chs)
        self.bn2 = bn2d(mid_chs)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_chs, reduce_chs=max(1, int(in_chs * 0.25)),
                                act_fn=self.act_fn, gate_fn=sigmoid)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, 1, bias=False)
        self.bn3 = bn2d(out_chs)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        return x


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, num_features=1280):
        super(EfficientNetB0, self).__init__()
        self.num_classes = num_classes
        self.act_fn = swish
        self.num_features = num_features
        margin = num_classes / (num_classes - 1) / 30
        margin *= 1 if num_classes <= num_features else num_features / num_classes
        scale = num_classes / (num_classes - 1) * math.log((num_classes - 1) * 0.9 / (1 - 0.9))

        bn2d = partial(nn.BatchNorm2d, momentum=0.1, eps=1e-05)

        self.conv_stem = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = bn2d(32)

        block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 16, bn2d)
        )
        block2 = nn.Sequential(
            InvertedResidual(16, 24, bn2d, dw_kernel_size=3, stride=2, drop_connect_rate=0.0125),
            InvertedResidual(24, 24, bn2d, dw_kernel_size=3, stride=1, drop_connect_rate=0.025)
        )
        block3 = nn.Sequential(
            InvertedResidual(24, 40, bn2d, dw_kernel_size=5, stride=2, drop_connect_rate=0.0375),
            InvertedResidual(40, 40, bn2d, dw_kernel_size=5, stride=1, drop_connect_rate=0.05)
        )
        block4 = nn.Sequential(
            InvertedResidual(40, 80, bn2d, dw_kernel_size=3, stride=2, drop_connect_rate=0.0625),
            InvertedResidual(80, 80, bn2d, dw_kernel_size=3, stride=1, drop_connect_rate=0.075),
            InvertedResidual(80, 80, bn2d, dw_kernel_size=3, stride=1, drop_connect_rate=0.0875)
        )
        block5 = nn.Sequential(
            InvertedResidual(80, 112, bn2d, dw_kernel_size=5, stride=1, drop_connect_rate=0.1),
            InvertedResidual(112, 112, bn2d, dw_kernel_size=5, stride=1, drop_connect_rate=0.1125),
            InvertedResidual(112, 112, bn2d, dw_kernel_size=5, stride=1, drop_connect_rate=0.125)
        )
        block6 = nn.Sequential(
            InvertedResidual(112, 192, bn2d, dw_kernel_size=5, stride=2, drop_connect_rate=0.1375),
            InvertedResidual(192, 192, bn2d, dw_kernel_size=5, stride=1, drop_connect_rate=0.15),
            InvertedResidual(192, 192, bn2d, dw_kernel_size=5, stride=1, drop_connect_rate=0.1625),
            InvertedResidual(192, 192, bn2d, dw_kernel_size=5, stride=1, drop_connect_rate=0.175)
        )
        block7 = nn.Sequential(
            InvertedResidual(192, 320, bn2d, dw_kernel_size=3, stride=1, drop_connect_rate=0.1875)
        )
        self.blocks = nn.Sequential(
            block1, block2, block3, block4, block5, block6, block7
        )

        self.conv_head = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = bn2d(1280)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ArcMarginModel(self.num_features, self.num_classes, margin=margin, scale=scale)

        for m in self.modules():
            self._initialize_weight_goog(m)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        del self.classifier
        if num_classes:
            margin = 0.5 * num_classes / (num_classes - 1)
            scale = num_classes / (num_classes - 1) * math.log((num_classes - 1) * 0.9 / (1 - 0.9))
            self.classifier = ArcMarginModel(self.num_features, num_classes, margin=margin, scale=scale)
        else:
            self.classifier = None

    def forward_features(self, x, pool=True):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x = self.blocks(x)

        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)
        if pool:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, input, target=None):
        x = self.forward_features(input)
        if self.classifier is not None:
            x = self.classifier(x, target)
        return x

    @staticmethod
    def _initialize_weight_goog(m):
        # weight init as per Tensorflow Official impl
        # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(0)  # fan-out
            init_range = 1.0 / math.sqrt(n)
            m.weight.data.uniform_(-init_range, init_range)
            m.bias.data.zero_()


def efficientnet_b0(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B0 """
    cfg = {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0-d6904d92.pth',
        'num_classes': 1000, 'first_conv': 'conv_stem', 'classifier': 'classifier'
    }
    model = EfficientNetB0(num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        load_pretrained(model, cfg, num_classes, in_chans)
    return model
