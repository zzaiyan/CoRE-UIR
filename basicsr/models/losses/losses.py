# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from lpips import LPIPS
try:
    from pytorch_msssim import MS_SSIM, SSIM
except ImportError:
    MS_SSIM = None
    SSIM = None

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class LPIPSLoss(nn.Module):
    
    def __init__(self, loss_weight=1.0, reduction='mean', net_type='vgg'):
        super(LPIPSLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.lpips = LPIPS(net=net_type)

    def forward(self, pred, target):
        # LPIPS expects input in range [-1, 1]
        # enable normalize to ensure the input is in the correct range
        loss = self.loss_weight * self.lpips(pred, target, normalize=True).mean()
        return loss, None


class _BaseSSIMLoss(nn.Module):

    metric_cls = None

    def __init__(self,
                 loss_weight=1.0,
                 window_size=11,
                 reduction='mean',
                 data_range=1.0,
                 window_sigma=1.5,
                 channel=3,
                 k=(0.01, 0.03),
                 metric_kwargs=None):
        super(_BaseSSIMLoss, self).__init__()
        if self.metric_cls is None:
            raise NotImplementedError('metric_cls must be defined in subclasses.')
        if reduction != 'mean':
            raise ValueError(f'{self.__class__.__name__} only supports mean reduction.')
        if window_size % 2 == 0:
            raise ValueError(f'{self.__class__.__name__} window_size must be odd.')

        self.loss_weight = loss_weight
        self.window_size = window_size
        self.reduction = reduction
        self.data_range = data_range
        self.window_sigma = window_sigma
        self.channel = channel
        self.k = k
        self.metric = self.metric_cls(
            data_range=self.data_range,
            size_average=True,
            win_size=self.window_size,
            win_sigma=self.window_sigma,
            channel=self.channel,
            spatial_dims=2,
            K=self.k,
            **(metric_kwargs or {}))

    def forward(self, pred, target):
        score = self.metric(pred, target)
        loss = 1 - score
        return self.loss_weight * loss


class SSIMLoss(_BaseSSIMLoss):

    metric_cls = SSIM

    def __init__(self,
                 loss_weight=1.0,
                 window_size=11,
                 reduction='mean',
                 data_range=1.0,
                 window_sigma=1.5,
                 channel=3,
                 k=(0.01, 0.03),
                 nonnegative_ssim=False):
        if SSIM is None:
            raise ImportError('pytorch_msssim is required to use SSIMLoss.')
        super(SSIMLoss, self).__init__(
            loss_weight=loss_weight,
            window_size=window_size,
            reduction=reduction,
            data_range=data_range,
            window_sigma=window_sigma,
            channel=channel,
            k=k,
            metric_kwargs={'nonnegative_ssim': nonnegative_ssim})


class MSSSIMLoss(_BaseSSIMLoss):

    metric_cls = MS_SSIM

    def __init__(self,
                 loss_weight=1.0,
                 window_size=11,
                 reduction='mean',
                 data_range=1.0,
                 window_sigma=1.5,
                 channel=3,
                 weights=None,
                 k=(0.01, 0.03)):
        if MS_SSIM is None:
            raise ImportError('pytorch_msssim is required to use MSSSIMLoss.')
        super(MSSSIMLoss, self).__init__(
            loss_weight=loss_weight,
            window_size=window_size,
            reduction=reduction,
            data_range=data_range,
            window_sigma=window_sigma,
            channel=channel,
            k=k,
            metric_kwargs={'weights': weights})
