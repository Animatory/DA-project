from functools import partial

import torch
import torch.nn.functional as F
# import torch_sparse
from torch import nn
from torch.nn import Parameter


class LinearVARD(nn.Module):
    """
    Dense layer implementation with weights VARD-prior (arxiv:1701.05369)
    """

    def __init__(self, in_features, out_features, bias=True, thresh=3, sparse_thresh=0.8, reg_weight=1e-4):
        super(LinearVARD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.thresh = thresh
        self.sparse_thresh = sparse_thresh
        self.reg_weight = reg_weight
        self.sparse_mult = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        if not self.training:
            # if self.sparse_mult:
            #     return torch_sparse.spmm(self.weight_indices, self.weight_values, self.out_features,
            #                              input.t()).t() + self.bias
            # else:
            return F.linear(input, self.weight_clipped, self.bias)

        W = self.weight
        mu = input.matmul(W.t())
        si = torch.sqrt((input * input).matmul(((torch.exp(self.log_alpha) * self.weight * self.weight) + 1e-8).t()))
        activation = mu + torch.normal(torch.zeros_like(mu), torch.ones_like(mu)) * si
        return activation + self.bias

    def reset_parameters(self):
        self.weight.data.normal_(std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(0, 0)
        self.log_sigma2.data.uniform_(-10, -10)

    @staticmethod
    def clip(tensor, to=8):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def get_clip_mask(self):
        log_alpha = self.clip(self.log_alpha)
        return torch.ge(log_alpha, self.thresh)

    def _update_sparse_weights(self):
        clip_mask = self.get_clip_mask()
        self.weight_indices = (clip_mask == 0).nonzero().t()
        self.weight_values = self.weight[~clip_mask]
        self.weight_clipped = torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)
        self.sparse_mult = (self.get_dropped_params_cnt() * 1.0 / torch.ones_like(
            self.weight).sum()) > self.sparse_thresh

    def train(self, mode=True):
        if not mode:
            self._update_sparse_weights()
        self.training = mode
        super(LinearVARD, self).train(mode)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.clip(self.log_alpha)
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return torch.exp(torch.log(-torch.sum(mdkl)) + torch.log(torch.tensor(self.reg_weight)))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        eps = 1e-8
        return self.log_sigma2 - torch.log(self.weight ** 2 + eps)


class Conv2dVARD(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, vard_init=-10, thresh=3, bias=True, reg_weight=1e-4):
        super(Conv2dVARD, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        if bias:
            bias = Parameter(torch.Tensor(out_channels))
        else:
            bias = None
        self.thresh = thresh
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vard_init = vard_init
        self.reg_weight = reg_weight
        self.log_sigma2 = Parameter(vard_init * torch.ones_like(self.weight))

    @staticmethod
    def clip(tensor, to=8):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        conv2d = partial(F.conv2d, stride=self.stride, padding=self.padding,
                         dilation=self.dilation, groups=self.groups)

        if not self.training:
            return conv2d(input, self.weights_clipped, bias=self.bias)

        W = self.weight
        conved_mu = conv2d(input, W)
        conved_si = torch.sqrt(1e-8 + conv2d(input * input, torch.exp(self.log_alpha) * W * W))
        conved = conved_mu + conved_si * torch.normal(torch.zeros_like(conved_mu), torch.ones_like(conved_mu))
        if self.bias is not None:
            conved = conved + self.bias[None, :, None, None]
        return conved

    def get_clip_mask(self):
        log_alpha = self.clip(self.log_alpha)
        return torch.ge(log_alpha, self.thresh)

    def _update_sparse_weights(self):
        clip_mask = self.get_clip_mask()
        self.weights_clipped = torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def train(self, mode=True):
        if not mode:
            self._update_sparse_weights()
        self.training = mode
        return super(Conv2dVARD, self).train(mode)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        log_alpha = self.clip(self.log_alpha)
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1
        return -torch.sum(mdkl) * torch.tensor(self.reg_weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        eps = 1e-8
        return self.log_sigma2 - torch.log(self.weight ** 2 + eps)


def get_vard_reg(module, reg=0):
    """
    :param module: model to evaluate variational dropout regularization for
    :param reg: auxiliary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, (LinearVARD, Conv2dVARD)):
        return reg + module.get_reg()
    if hasattr(module, 'children'):
        return reg + sum([get_vard_reg(submodule) for submodule in module.children()])
    return reg


def _get_dropped_params_cnt(module, cnt=0):
    if hasattr(module, 'get_dropped_params_cnt'):
        return cnt + module.get_dropped_params_cnt()
    if hasattr(module, 'children'):
        return cnt + sum([_get_dropped_params_cnt(submodule) for submodule in module.children()])
    return cnt


def _get_params_cnt(module, cnt=0):
    if isinstance(module, (LinearVARD, Conv2dVARD)):
        return cnt + module.weight.nelement()
    if hasattr(module, 'children'):
        return cnt + sum([_get_params_cnt(submodule) for submodule in module.children()])
    return cnt + sum(p.numel() for p in module.parameters())


def get_dropped_params_ratio(model):
    return _get_dropped_params_cnt(model) * 1.0 / _get_params_cnt(model)


def update_weights_reg(module, epoch, f):
    if isinstance(module, (LinearVARD, Conv2dVARD)):
        module.reg_weight = f(epoch)
    if hasattr(module, 'children'):
        [get_vard_reg(submodule) for submodule in module.children()]