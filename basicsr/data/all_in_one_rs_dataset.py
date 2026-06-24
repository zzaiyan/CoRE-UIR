import math
import os
import random
from os import path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from prior.utils_image import (
    PriorViewTransformMixin,
    resolve_router_view_config,
)


IMG_EXTENSIONS = {
    '.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.ppm',
    '.pgm', '.pbm', '.tif', '.tiff', '.webp'
}


def read_img(filename, to_float=False):
    """Read an RGB image from disk."""
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {filename}')
    if to_float:
        img = img.astype('float32') / 255.0
    return img[:, :, ::-1]


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def rs_augment(imgs=None, size=256, edge_decay=0., data_augment=True):
    """Crop and augment a list of aligned HWC images."""
    imgs = imgs or []
    h, w, _ = imgs[0].shape
    crop_h, crop_w = size, size

    if random.random() < crop_h / h * edge_decay:
        hs = 0 if random.randint(0, 1) == 0 else h - crop_h
    else:
        hs = random.randint(0, h - crop_h)

    if random.random() < crop_w / w * edge_decay:
        ws = 0 if random.randint(0, 1) == 0 else w - crop_w
    else:
        ws = random.randint(0, w - crop_w)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][hs:(hs + crop_h), ws:(ws + crop_w), :]

    if data_augment:
        if random.randint(0, 1) == 1:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1)

        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def rs_align(imgs=None, size=256):
    """Center crop a list of aligned HWC images."""
    imgs = imgs or []
    h, w, _ = imgs[0].shape
    crop_h, crop_w = size, size

    hs = (h - crop_h) // 2
    ws = (w - crop_w) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][hs:(hs + crop_h), ws:(ws + crop_w), :]

    return imgs


def _is_image_file(filename):
    return osp.splitext(filename)[1].lower() in IMG_EXTENSIONS


def _sorted_image_paths(folder):
    if not osp.isdir(folder):
        raise FileNotFoundError(f'Image folder not found: {folder}')

    paths = []
    for filename in sorted(os.listdir(folder)):
        full_path = osp.join(folder, filename)
        if osp.isfile(full_path) and _is_image_file(filename):
            paths.append(full_path)

    if not paths:
        raise ValueError(f'No images found in folder: {folder}')
    return paths


def _build_stem_to_path(folder):
    stem_to_path = {}
    for path in _sorted_image_paths(folder):
        stem = osp.splitext(osp.basename(path))[0]
        if stem in stem_to_path:
            raise ValueError(f'Duplicate image stem "{stem}" in folder: {folder}')
        stem_to_path[stem] = path
    return stem_to_path


def _linearly_select_items(items, ratio):
    if not 0 < ratio <= 1:
        raise ValueError(f'select_ratio must be in (0, 1], but got {ratio}')

    items = list(items)
    if ratio >= 1 or len(items) <= 1:
        return items

    target_count = max(1, int(math.ceil(len(items) * ratio)))
    indices = np.linspace(0, len(items) - 1, num=target_count, dtype=int).tolist()
    return [items[i] for i in indices]


def _normalize_repeat(value):
    if value is None:
        return 1
    if isinstance(value, bool):
        raise TypeError('repeat must be an integer greater than or equal to 1.')

    try:
        repeat_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError('repeat must be an integer greater than or equal to 1.') from exc

    repeat_int = int(repeat_value)
    if repeat_int < 1 or repeat_value != repeat_int:
        raise ValueError('repeat must be an integer greater than or equal to 1.')
    return repeat_int


def _infer_degrade_types(lq_folders):
    return [osp.basename(osp.normpath(folder)) for folder in lq_folders]


class _AllInOneBaseDataset(PriorViewTransformMixin, data.Dataset):
    """Shared logic for all-in-one restoration datasets."""

    def __init__(self, opt):
        super().__init__()
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.clip_transform_mode = opt.get('clip_transform_mode', 'inference')
        self.size = opt.get('gt_size', 256)
        self.edge_decay = opt.get('edge_decay', 0.)
        self.data_augment = opt.get('data_augment', True)
        self.cache_memory = opt.get('cache_memory', False)
        self.max_samples = opt.get('max_samples')
        self.select_ratio = float(opt.get('select_ratio', 1.0))
        self.repeat = _normalize_repeat(opt.get('repeat', 1))
        self.view_mode, self.view_size = resolve_router_view_config(
            view_mode=opt.get('view_mode'),
            view_size=opt.get('view_size'),
            default_view_mode='dual',
            default_view_size=224)

        self.cached_gt_files = {}
        self.cached_lq_files = {}

        self.degrade_types = self._init_degrade_types(opt)
        self.samples_by_task = self._prepare_task_samples()
        self.num_tasks = len(self.samples_by_task)

        if self.num_tasks == 0:
            raise ValueError('No degradation tasks were created.')

        if any(len(samples) == 0 for samples in self.samples_by_task):
            raise ValueError('At least one degradation task has no valid samples.')

        self.max_task_num = min(len(samples) for samples in self.samples_by_task)
        self.total_num = sum(len(samples) for samples in self.samples_by_task)
        self.samples = [
            sample for task_samples in self.samples_by_task for sample in task_samples
        ]

    def _init_degrade_types(self, opt):
        degrade_types = opt.get('degrade_type')
        if degrade_types is None:
            degrade_types = self._default_degrade_types(opt)
        if not isinstance(degrade_types, list) or not degrade_types:
            raise ValueError('degrade_type must be a non-empty list of strings.')
        return degrade_types

    def _default_degrade_types(self, opt):
        raise NotImplementedError

    def _prepare_task_samples(self):
        raise NotImplementedError

    def _apply_subset(self, items):
        items = _linearly_select_items(items, self.select_ratio)
        if self.max_samples is not None:
            items = items[:self.max_samples]
        if self.repeat > 1:
            items = items * self.repeat
        return items

    def _load_image(self, path, cache_dict):
        if self.cache_memory and path in cache_dict:
            return cache_dict[path]

        img = read_img(path, to_float=True)
        if self.cache_memory:
            cache_dict[path] = img
        return img

    def _build_clip_tensor(self, img_lq):
        clip_image = Image.fromarray(np.clip(img_lq * 255., 0, 255).astype(np.uint8))
        phase = 'train' if self.phase == 'train' and self.clip_transform_mode == 'train' else 'inference'
        return self.build_router_views(
            clip_image,
            view_mode=self.view_mode,
            view_size=self.view_size,
            phase=phase)

    def __getitem__(self, index):
        sample = self.samples[index]

        gt_path = sample['gt_path']
        lq_path = sample['lq_path']

        img_gt = self._load_image(gt_path, self.cached_gt_files)
        img_lq = self._load_image(lq_path, self.cached_lq_files)
        img_clip = self._build_clip_tensor(img_lq)

        if self.phase == 'train':
            img_gt, img_lq = rs_augment(
                [img_gt, img_lq], self.size, self.edge_decay, self.data_augment)
        elif self.phase in ('val', 'valid'):
            img_gt, img_lq = rs_align([img_gt, img_lq], self.size)

        img_gt = torch.from_numpy(hwc_to_chw(img_gt)).float()
        img_lq = torch.from_numpy(hwc_to_chw(img_lq)).float()

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_clip': img_clip,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'degrade_type': sample['degrade_type'],
            'filename': sample['filename']
        }

    def __len__(self):
        return self.total_num


class AllInOneRSDataset(_AllInOneBaseDataset):
    """All-in-one dataset with one shared GT folder and multiple LQ folders."""

    def _default_degrade_types(self, opt):
        return _infer_degrade_types(opt['dataroot_lq'])

    def _prepare_task_samples(self):
        self.gt_folder = self.opt['dataroot_gt']
        self.lq_folders = self.opt['dataroot_lq']

        if len(self.lq_folders) != len(self.degrade_types):
            raise ValueError('dataroot_lq and degrade_type must have the same length.')

        gt_map = _build_stem_to_path(self.gt_folder)
        common_stems = set(gt_map.keys())
        lq_maps = []
        for lq_folder in self.lq_folders:
            lq_map = _build_stem_to_path(lq_folder)
            lq_maps.append(lq_map)
            common_stems &= set(lq_map.keys())

        common_stems = sorted(common_stems)
        if not common_stems:
            raise ValueError('No matched image stems were found across shared GT and LQ folders.')

        common_stems = self._apply_subset(common_stems)

        task_samples = []
        for degrade_type, lq_map in zip(self.degrade_types, lq_maps):
            samples = []
            for stem in common_stems:
                lq_path = lq_map[stem]
                gt_path = gt_map[stem]
                samples.append({
                    'degrade_type': degrade_type,
                    'filename': osp.splitext(osp.basename(lq_path))[0],
                    'lq_path': lq_path,
                    'gt_path': gt_path
                })
            task_samples.append(samples)

        return task_samples


class AllInOnePairDataset(_AllInOneBaseDataset):
    """All-in-one dataset where each degradation has its own GT and LQ folders."""

    def _default_degrade_types(self, opt):
        return _infer_degrade_types(opt['dataroot_lq'])

    def _prepare_task_samples(self):
        self.gt_folders = self.opt['dataroot_gt']
        self.lq_folders = self.opt['dataroot_lq']

        if not isinstance(self.gt_folders, list) or not isinstance(self.lq_folders, list):
            raise ValueError('AllInOnePairDataset expects list-style dataroot_gt and dataroot_lq.')
        if len(self.gt_folders) != len(self.degrade_types) or len(self.lq_folders) != len(self.degrade_types):
            raise ValueError('degrade_type, dataroot_gt and dataroot_lq must have the same length.')

        task_samples = []
        for degrade_type, gt_folder, lq_folder in zip(
                self.degrade_types, self.gt_folders, self.lq_folders):
            gt_map = _build_stem_to_path(gt_folder)
            lq_map = _build_stem_to_path(lq_folder)
            common_stems = sorted(set(gt_map.keys()) & set(lq_map.keys()))

            if not common_stems:
                raise ValueError(
                    f'No matched image stems were found for degradation "{degrade_type}".')

            common_stems = self._apply_subset(common_stems)

            samples = []
            for stem in common_stems:
                lq_path = lq_map[stem]
                gt_path = gt_map[stem]
                samples.append({
                    'degrade_type': degrade_type,
                    'filename': osp.splitext(osp.basename(lq_path))[0],
                    'lq_path': lq_path,
                    'gt_path': gt_path
                })
            task_samples.append(samples)

        return task_samples
