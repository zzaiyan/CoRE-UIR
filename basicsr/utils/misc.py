# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import numpy as np
import os
import random
import time
import torch
from os import path as osp

from .dist_util import master_only
from .logger import get_root_logger


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if path is None:
            continue
        if ('strict_load' not in key) and ('pretrain' not in key) and ('resume' not in key):
            os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def scandir_SIDD(dir_path, keywords=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        keywords (str | tuple(str), optional): File keywords that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (keywords is not None) and not isinstance(keywords, (str, tuple)):
        raise TypeError('"keywords" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, keywords, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if keywords is None:
                    yield return_path
                elif return_path.find(keywords) > 0:
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, keywords=keywords, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, keywords=keywords, recursive=recursive)


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning(
                'pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (
                    basename not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = osp.join(
                    opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formated file siz.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} {suffix}'


class RunningAverage:
    """在线计算平均值的类，用于实时显示验证指标的均值。

    支持多个指标同时计算，每次更新都会重新计算平均值。
    """

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, **kwargs):
        """更新指标值。

        Args:
            **kwargs: 指标名称和对应的值，例如 psnr=30.5, ssim=0.85
        """
        for metric_name, value in kwargs.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = 0.0
                self.counts[metric_name] = 0

            self.counts[metric_name] += 1
            # 在线更新平均值: new_avg = old_avg + (new_value - old_avg) / count
            self.metrics[metric_name] += (value -
                                          self.metrics[metric_name]) / self.counts[metric_name]

    def get_averages(self):
        """获取所有指标的平均值。

        Returns:
            dict: 包含所有指标平均值的字典
        """
        return self.metrics.copy()

    def get_average(self, metric_name):
        """获取指定指标的平均值。

        Args:
            metric_name (str): 指标名称

        Returns:
            float: 指标的平均值，如果不存在则返回0.0
        """
        return self.metrics.get(metric_name, 0.0)

    def get_count(self, metric_name=None):
        """获取样本数量。

        Args:
            metric_name (str, optional): 指标名称。如果为None，返回第一个指标的计数

        Returns:
            int: 样本数量
        """
        if metric_name is None:
            return list(self.counts.values())[0] if self.counts else 0
        return self.counts.get(metric_name, 0)

    def format_string(self, precision=4):
        """格式化输出字符串，用于进度条显示。

        Args:
            precision (int): 小数点后保留位数，默认为4

        Returns:
            str: 格式化的字符串，例如 "PSNR: 30.5000, SSIM: 0.8500"
        """
        if not self.metrics:
            return "No metrics"

        format_str = f"{{:.{precision}f}}"
        parts = []
        for metric_name, avg_value in self.metrics.items():
            parts.append(
                f"{metric_name.upper()}: {format_str.format(avg_value)}")

        return ", ".join(parts)

    def reset(self):
        """重置所有指标。"""
        self.metrics.clear()
        self.counts.clear()
