import math

import torch
from torch import nn
from torch.nn import functional as F
from models.effnet import ArcMarginModel
from models.utils import Swish


class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.num_features = 128
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2, 2)),
            nn.Dropout()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2, 2)),
            nn.Dropout()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.fc4 = nn.Linear(self.num_features, self.num_features)

        margin = num_classes / (num_classes - 1) / 30
        margin *= 1 if num_classes <= self.num_features else self.num_features / num_classes
        scale = num_classes / (num_classes - 1) * math.log((num_classes - 1) * 0.9 / (1 - 0.9))
        self.classifier_arc_margin = ArcMarginModel(self.num_features, self.num_classes, margin=margin, scale=scale)

        self.classifier = nn.Linear(self.num_features, num_classes)

        # for m in self.modules():
        #     self._initialize_weight_goog(m)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)
        x = self.fc4(x)
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

    def forward_logits_and_features(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits, features


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


class ContentBCE(nn.Module):
    def __init__(self, confidence_thresh=0.96837722):
        super().__init__()
        self.confidence_thresh = confidence_thresh

    @staticmethod
    def robust_binary_crossentropy(pred, tgt):
        inv_tgt = -tgt + 1.0
        inv_pred = -pred + 1.0 + 1e-6
        return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

    def forward(self, stu_out, tea_out):
        conf_tea = torch.max(tea_out, 1)[0]
        unsup_mask = (conf_tea > self.confidence_thresh).float()

        aug_loss = self.robust_binary_crossentropy(stu_out, tea_out)
        aug_loss = aug_loss.mean(dim=1)
        unsup_loss = (aug_loss * unsup_mask).mean()

        return unsup_loss


class EMAWeightOptimizer:
    """
    Exponential moving average weight optimizer for mean teacher model
    """

    def __init__(self, target_net, source_net, alpha=0.999):
        self.target_params = list(target_net.parameters())
        self.source_params = list(source_net.parameters())
        self.alpha = alpha

        for p, src_p in zip(self.target_params, self.source_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.target_params, self.source_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
