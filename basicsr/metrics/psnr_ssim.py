# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# modified from https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from math import exp

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from skimage.metrics import structural_similarity


# ------------------------------------------------------------------------
# PSNR
# ------------------------------------------------------------------------
def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    
    Args:
        img1 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        img2 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Deprecated. Now always calculates on all channels.
            Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    
    # print(f"Get type: {type(img1)}, shape: {img1.shape}")

    is_tensor = torch.is_tensor(img1)
    assert is_tensor, 'Now only support tensor inputs.'
    if is_tensor:
        # Move to cuda if available and not already on cuda
        if torch.cuda.is_available():
            device = torch.device('cuda')
            img1 = img1.to(device)
            img2 = img2.to(device)

        # Clone tensors to avoid modifying original data
        img1 = img1.clone().to(dtype=torch.float32)
        img2 = img2.clone().to(dtype=torch.float32)
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
            img2 = img2.squeeze(0)
    else:
        if input_order not in ['HWC', 'CHW']:
            raise ValueError(
                f'Wrong input_order {input_order}. Supported input_orders are '
                '"HWC" and "CHW"')
        img1 = reorder_image(img1, input_order=input_order)
        img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        if is_tensor:
            # Tensor is CHW
            img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
        else:
            # Numpy is HWC
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if is_tensor:
        # Use torch.mse_loss for consistency with Ada4DIR
        mse = F.mse_loss(img1, img2)
        # Determine max value from tensor range
        max_value = 1. if torch.max(img2) <= 1. else 255.
    else:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        mse = np.mean((img1 - img2)**2)
        max_value = 1. if img2.max() <= 1. else 255.

    if mse == 0:
        return float('inf')
        
    if is_tensor:
        return 10. * torch.log10(max_value / mse).item()
    else:
        return 10. * np.log10(max_value / mse)


# ------------------------------------------------------------------------
# SSIM (PyTorch implementation for consistency with Ada4DIR)
# ------------------------------------------------------------------------

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _create_window(window_size, channel=1):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim_pytorch(img1, img2, window_size=11, val_range=None, size_average=True):
    """Pytorch implementation of SSIM, consistent with Ada4DIR."""
    # Move to cuda if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        img1 = img1.to(device)
        img2 = img2.to(device)
        
    img1 = img1.to(dtype=torch.float32)
    img2 = img2.to(dtype=torch.float32)

    if val_range is None:
        if torch.max(img2) > 10:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img2) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    window = _create_window(window_size, channel=channel).to(img1.device).to(dtype=torch.float32)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False,
                   **kwargs):
    """Calculate SSIM (structural similarity).

    This function is now a wrapper for the PyTorch-based implementation to ensure
    consistency with the Ada4DIR project. The `test_y_channel` and `ssim3d`
    arguments are ignored.

    Args:
        img1 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        img2 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Deprecated. Ignored.
        kwargs: Other arguments are ignored.

    Returns:print(f"Get type: {type(img1)}, shape: {img1.shape}")
        float: ssim result.
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    is_tensor = torch.is_tensor(img1)
    
    # print(f"Get type: {type(img1)}, shape: {img1.shape}")
    assert is_tensor, 'Now only support tensor inputs.'

    # Convert to torch.Tensor if input is numpy array
    if not is_tensor:
        img1 = torch.from_numpy(reorder_image(img1, input_order=input_order).transpose(2, 0, 1)).float()
        img2 = torch.from_numpy(reorder_image(img2, input_order=input_order).transpose(2, 0, 1)).float()

    # Add batch dimension if not present
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)

    # Crop border
    if crop_border != 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    return _ssim_pytorch(img1, img2)


# ------------------------------------------------------------------------
# Additional left-image helpers kept for stereo-style result compatibility.
# ------------------------------------------------------------------------

def calculate_psnr_left(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    assert input_order == 'HWC'
    assert crop_border == 0

    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_psnr(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel)

def _ssim(img1, img2, max_value):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * max_value)**2
    C2 = (0.03 * max_value)**2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim_left(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False,
                   ssim3d=True):
    assert input_order == 'HWC'
    assert crop_border == 0

    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_ssim(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel)

def calculate_skimage_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel=True)

def calculate_skimage_ssim_left(img1, img2):
    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_skimage_ssim(img1=img1, img2=img2)
