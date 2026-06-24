from os import path as osp

import torch
from torchvision.transforms.functional import normalize

from basicsr.data.all_in_one_rs_dataset import (
    _AllInOneBaseDataset,
    _infer_degrade_types,
    _sorted_image_paths,
    hwc_to_chw,
)


class SingleRSDataset(_AllInOneBaseDataset):
    """No-reference remote-sensing dataset for result export only.

    The dataset follows the multi-folder layout of `AllInOneRSDataset` on the
    degraded-image side, but it does not require GT images. It is intended for
    test-time export on real-world inputs.
    """

    has_gt = False

    def __init__(self, opt):
        super().__init__(opt)
        if self.phase == 'train':
            raise ValueError(
                'SingleRSDataset is for val/test only because it has no '
                'ground-truth supervision.')

    def _default_degrade_types(self, opt):
        return _infer_degrade_types(opt['dataroot_lq'])

    def _prepare_task_samples(self):
        self.lq_folders = self.opt['dataroot_lq']
        if not isinstance(self.lq_folders, list):
            raise ValueError('SingleRSDataset expects list-style dataroot_lq.')
        if len(self.lq_folders) != len(self.degrade_types):
            raise ValueError('dataroot_lq and degrade_type must have the same length.')

        task_samples = []
        for degrade_type, lq_folder in zip(self.degrade_types, self.lq_folders):
            lq_paths = self._apply_subset(_sorted_image_paths(lq_folder))
            samples = []
            for lq_path in lq_paths:
                samples.append({
                    'degrade_type': degrade_type,
                    'filename': osp.splitext(osp.basename(lq_path))[0],
                    'lq_path': lq_path,
                })
            task_samples.append(samples)
        return task_samples

    def __getitem__(self, index):
        sample = self.samples[index]
        lq_path = sample['lq_path']

        img_lq = self._load_image(lq_path, self.cached_lq_files)
        img_clip = self._build_clip_tensor(img_lq)
        img_lq = torch.from_numpy(hwc_to_chw(img_lq)).float()

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'lq_clip': img_clip,
            'lq_path': lq_path,
            'degrade_type': sample['degrade_type'],
            'filename': sample['filename']
        }
