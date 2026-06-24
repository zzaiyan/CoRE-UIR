# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging
import os
import math
import pickle
import json
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from os import path as osp
from tqdm import tqdm
import importlib

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.dist_util import master_only, get_dist_info
from basicsr.utils import get_root_logger, imwrite, tensor2img

metric_module = importlib.import_module('basicsr.metrics')

logger = logging.getLogger('basicsr')


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.lq = None
        self.lq_clip = None
        self.gt = None
        self.output = None

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass
    
    def free_gpu_memory(self):
        self.lq = None
        self.lq_clip = None
        self.gt = None
        self.output = None
        torch.cuda.empty_cache()

    def _torch_load_compat(self, load_path, map_location='cpu', weights_only=True):
        """Load checkpoints across PyTorch versions and Lightning-style files.

        PyTorch 2.5 defaults to a stricter ``weights_only=True`` unpickler that
        rejects some trusted training checkpoints, such as Lightning-produced
        router checkpoints containing ``AttributeDict`` metadata. We first try
        the safe path and only fall back to ``weights_only=False`` for local
        project checkpoints when the new unpickler explicitly rejects them.
        """
        try:
            return torch.load(load_path, map_location=map_location, weights_only=weights_only)
        except TypeError:
            if not weights_only:
                raise
            return torch.load(load_path, map_location=map_location)
        except pickle.UnpicklingError as exc:
            if not weights_only or 'Weights only load failed' not in str(exc):
                raise
            logger.warning(
                'weights_only=True rejected checkpoint %s; falling back to '
                'weights_only=False for trusted local loading.', load_path)
            return torch.load(load_path, map_location=map_location, weights_only=False)

    def _get_batch_value(self, data, key, index, default=None):
        value = data.get(key, default)
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[index]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value[index]
        return value

    def _get_sample_identity(self, data, index):
        lq_path = self._get_batch_value(data, 'lq_path', index)
        gt_path = self._get_batch_value(data, 'gt_path', index)
        filename = self._get_batch_value(data, 'filename', index)
        deg_type = self._get_batch_value(data, 'degrade_type', index)

        if deg_type is None and lq_path is not None:
            deg_type = osp.basename(osp.dirname(osp.normpath(str(lq_path))))
        if not deg_type:
            deg_type = 'unknown'

        if filename is None:
            ref_path = lq_path or gt_path
            if ref_path is not None:
                filename = osp.splitext(osp.basename(str(ref_path)))[0]
            else:
                filename = f'sample_{index:06d}'

        img_name = f'{deg_type}/{filename}'
        return img_name, deg_type

    def _prepare_validation_metric_items(self):
        """Resolve metric configs once before a validation loop.

        Returns:
            list[tuple[str, callable, dict]]: Tuples of metric name, metric
            function, and keyword arguments without the `type` field.
        """
        metric_items = []
        for name, opt_ in self.opt['val']['metrics'].items():
            metric_cfg = dict(opt_)
            metric_type = metric_cfg.pop('type')
            metric_items.append((name, getattr(metric_module, metric_type), metric_cfg))
        return metric_items

    def _should_compute_validation_metrics(self, dataloader):
        """Return whether metrics can be evaluated for the given dataset."""
        metrics_opt = self.opt['val'].get('metrics')
        dataset_has_gt = getattr(dataloader.dataset, 'has_gt', True)
        if metrics_opt is not None and not dataset_has_gt:
            logger.info(
                'Validation metrics are disabled for dataset %s because it has '
                'no ground-truth images.', dataloader.dataset.opt['name'])
        return metrics_opt is not None and dataset_has_gt

    def _get_visualization_root(self, dataset_name):
        """Return the root directory used to export images for a dataset split."""
        if self.opt['is_train']:
            return self.opt['path']['visualization']
        return osp.join(self.opt['path']['visualization'], dataset_name)

    def _get_save_image_path(self, dataset_name, img_name):
        """Return the output path for one validation/test image."""
        return osp.join(self._get_visualization_root(dataset_name), f'{img_name}.png')

    def _get_grid_source_tensor(self):
        """Return the tensor that should be spatially split for grid testing."""
        if self.lq is None:
            raise AttributeError('grids() expects the model to provide self.lq.')
        return self.lq

    def _set_grid_source_tensor(self, value):
        """Update the tensor that backs grid-based inference."""
        self.lq = value

    def _get_grid_reference_tensor(self):
        """Return the tensor whose spatial size defines the reconstruction canvas."""
        if self.gt is not None:
            return self.gt
        return self._get_grid_source_tensor()

    def _get_grid_crop_size(self, h, w):
        """Compute the grid crop size aligned to the model scale factor."""
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val']['crop_size_w']
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h = crop_size_h // self.scale * self.scale
        crop_size_w = crop_size_w // self.scale * self.scale
        return crop_size_h, crop_size_w

    def grids(self):
        """Split the current validation sample into overlapping inference tiles.

        The method stores the original source tensor, replaces it with the
        concatenated tiles, and records tile coordinates in `self.idxes`.
        `test()` is then expected to process the tiled tensor and populate
        `self.outs` when `val.grids: true`.
        """
        ref_tensor = self._get_grid_reference_tensor()
        source_tensor = self._get_grid_source_tensor()
        b, c, h, w = ref_tensor.size()
        self.original_size = (b, c, h, w)

        assert b == 1

        crop_size_h, crop_size_w = self._get_grid_crop_size(h, w)
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        step_j = crop_size_w if num_col == 1 else math.ceil(
            (w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil(
            (h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(source_tensor[:, :, i // scale:(i + crop_size_h) // scale,
                                           j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self._grid_origin_tensor = source_tensor
        self._set_grid_source_tensor(torch.cat(parts, dim=0))
        self.idxes = idxes

    def grids_inverse(self):
        """Merge tiled predictions from `self.outs` back into full-resolution output.

        The merged prediction is averaged in overlapping regions, stored in
        `self.output`, and the original source tensor is restored.
        """
        outs = getattr(self, 'outs', None)
        if outs is None or not outs:
            raise AttributeError('grids_inverse() expects self.outs to be populated in test().')
        grid_origin_tensor = getattr(self, '_grid_origin_tensor', None)
        if grid_origin_tensor is None:
            raise AttributeError('grids_inverse() expects self._grid_origin_tensor to be populated in grids().')

        first_chunk = outs[0]
        out_device = first_chunk.device
        out_dtype = first_chunk.dtype
        preds = torch.zeros(self.original_size, device=out_device, dtype=out_dtype)
        b, _, h, w = self.original_size
        count_mt = torch.zeros((b, 1, h, w), device=out_device, dtype=out_dtype)

        crop_size_h, crop_size_w = self._get_grid_crop_size(h, w)

        flat_outs = []
        for chunk in outs:
            if chunk.dim() == 4:
                flat_outs.extend(chunk.unbind(0))
            else:
                flat_outs.append(chunk)

        for pred, each_idx in zip(flat_outs, self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + crop_size_h, j:j + crop_size_w] += pred
            count_mt[0, 0, i:i + crop_size_h, j:j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self._set_grid_source_tensor(grid_origin_tensor)

        self.outs = None
        self._grid_origin_tensor = None

    def validation(self, dataloader, current_iter, tb_logger, save_img=False, rgb2bgr=True, use_image=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
            rgb2bgr (bool): Whether to save images using rgb2bgr. Default: True
            use_image (bool): Whether to use saved images to compute metrics (PSNR, SSIM), if not, then use data directly from network' output. Default: True
        """
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return self.nondist_validation(dataloader, current_iter, tb_logger,
                                    save_img, rgb2bgr, use_image)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        self.free_gpu_memory()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self._should_compute_validation_metrics(dataloader)
        if with_metrics:
            metric_names = list(self.opt['val']['metrics'].keys())
            metric_items = self._prepare_validation_metric_items()
            self.metric_results = {
                metric: 0
                for metric in metric_names
            }
            self.metric_results_per_type = {}
        else:
            metric_names = []
            metric_items = []

        rank, world_size = get_dist_info()
        
        if rank == 0:
            pbar = tqdm(total=len(dataloader.dataset), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            result_tensor = self.output
            gt_tensor = self.gt if self.gt is not None else None

            cur_batch = result_tensor.shape[0]
            for b in range(cur_batch):
                img_name, deg_type = self._get_sample_identity(val_data, b)

                result_b = result_tensor[b].detach()
                gt_b = gt_tensor[b].detach() if gt_tensor is not None else None
                save_img_path = None

                if save_img:
                    sr_img = tensor2img([result_b], rgb2bgr=rgb2bgr)
                    if sr_img.shape[2] == 6:
                        L_img = sr_img[:, :, :3]
                        R_img = sr_img[:, :, 3:]

                        visual_dir = osp.join(
                            self.opt['path']['visualization'], dataset_name)

                        imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    else:
                        save_img_path = self._get_save_image_path(dataset_name, img_name)
                        imwrite(sr_img, save_img_path)
                if with_metrics:
                    if deg_type not in self.metric_results_per_type:
                        self.metric_results_per_type[deg_type] = {
                            metric: 0 for metric in metric_names
                        }
                        self.metric_results_per_type[deg_type]['cnt'] = 0
                    self.metric_results_per_type[deg_type]['cnt'] += 1

                    for name, metric_fn, metric_cfg in metric_items:
                        val = metric_fn(result_b, gt_b, **metric_cfg)
                        self.metric_results[name] += val
                        self.metric_results_per_type[deg_type][name] += val

                cnt += 1

            if rank == 0:
                pbar.update(cur_batch * world_size)
                pbar.set_description(f'Test {img_name}')
                pbar_log = {}
                if with_metrics:
                    all_metrics = []
                    for m_key in metric_names:
                        if m_key in self.metric_results:
                            val = self.metric_results[m_key] / cnt
                            fmt_val = f"{val:.2f}" if val > 1 else f"{val:.4f}"
                            all_metrics.append(fmt_val)
                    pbar_log['all'] = f"({'/'.join(all_metrics)})"

                    for deg_type, type_metrics in self.metric_results_per_type.items():
                        cnt_type = type_metrics['cnt']
                        if cnt_type > 0:
                            metric_values = []
                            for m_key in metric_names:
                                if m_key in type_metrics:
                                    val = type_metrics[m_key]/cnt_type
                                    fmt_val = f"{val:.2f}" if val > 1 else f"{val:.4f}"
                                    metric_values.append(fmt_val)
                            pbar_log[deg_type] = f"({'/'.join(metric_values)})"
                if pbar_log:
                    pbar.set_postfix(**pbar_log)
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        if with_metrics:
            collected_metrics = OrderedDict()
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(
                    self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(
                cnt).float().to(self.device)
            self.collected_metrics = collected_metrics
            keys = []
            metrics = []
            for name, value in self.collected_metrics.items():
                keys.append(name)
                metrics.append(value)
            metrics = torch.stack(metrics, 0)
            torch.distributed.reduce(metrics, dst=0)

            gathered_per_type = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_per_type, self.metric_results_per_type)
            if self.opt['rank'] == 0:
                metrics_dict = {}
                cnt = 0
                for key, metric in zip(keys, metrics):
                    if key == 'cnt':
                        cnt = float(metric)
                        continue
                    metrics_dict[key] = float(metric)

                for key in metrics_dict:
                    metrics_dict[key] /= cnt

                merged_per_type = {}
                for data in gathered_per_type:
                    for deg_type, type_metrics in data.items():
                        if deg_type not in merged_per_type:
                            merged_per_type[deg_type] = {k: 0.0 for k in type_metrics}
                        for k, v in type_metrics.items():
                            merged_per_type[deg_type][k] += v
                
                for deg_type, type_metrics in merged_per_type.items():
                    cnt_type = type_metrics.pop('cnt', 0)
                    if cnt_type > 0:
                        for m_name, m_val in type_metrics.items():
                            metrics_dict[f'{deg_type}/{m_name}'] = m_val / cnt_type

                self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                                   tb_logger, metrics_dict)
        self.free_gpu_memory()
        return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        self.free_gpu_memory()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self._should_compute_validation_metrics(dataloader)
        if with_metrics:
            metric_names = list(self.opt['val']['metrics'].keys())
            metric_items = self._prepare_validation_metric_items()
            self.metric_results = {
                metric: 0
                for metric in metric_names
            }
            self.metric_results_per_type = {}
        else:
            metric_names = []
            metric_items = []

        pbar = tqdm(total=len(dataloader.dataset), unit='image')
        cnt = 0

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            result_tensor = self.output
            gt_tensor = self.gt if self.gt is not None else None

            cur_batch = result_tensor.shape[0]
            for b in range(cur_batch):
                img_name, deg_type = self._get_sample_identity(val_data, b)

                result_b = result_tensor[b].detach()
                gt_b = gt_tensor[b].detach() if gt_tensor is not None else None
                save_img_path = None

                if save_img:
                    sr_img = tensor2img([result_b], rgb2bgr=rgb2bgr)
                    if sr_img.shape[2] == 6:
                        L_img = sr_img[:, :, :3]
                        R_img = sr_img[:, :, 3:]

                        visual_dir = osp.join(
                            self.opt['path']['visualization'], dataset_name)

                        imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    else:
                        save_img_path = self._get_save_image_path(dataset_name, img_name)
                        imwrite(sr_img, save_img_path)
                if with_metrics:
                    if deg_type not in self.metric_results_per_type:
                        self.metric_results_per_type[deg_type] = {
                            metric: 0 for metric in metric_names
                        }
                        self.metric_results_per_type[deg_type]['cnt'] = 0
                    self.metric_results_per_type[deg_type]['cnt'] += 1
                    
                    # if save_img:
                    #     print('\nComputing metrics for', img_name, end=': ')

                    for name, metric_fn, metric_cfg in metric_items:
                        val = metric_fn(result_b, gt_b, **metric_cfg)
                        
                        # if save_img:
                        #     print(f'{name}={val:.4f}', end='; ')
                        
                        self.metric_results[name] += val
                        self.metric_results_per_type[deg_type][name] += val

                cnt += 1
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
                if with_metrics:
                    pbar_log = {}
                    all_metrics = []
                    for m_key in metric_names:
                        if m_key in self.metric_results:
                            val = self.metric_results[m_key] / cnt
                            fmt_val = f"{val:.2f}" if val > 1 else f"{val:.4f}"
                            all_metrics.append(fmt_val)
                    pbar_log['all'] = f"({'/'.join(all_metrics)})"

                    for deg_type, type_metrics in self.metric_results_per_type.items():
                        cnt_type = type_metrics['cnt']
                        if cnt_type > 0:
                            metric_values = []
                            for m_key in metric_names:
                                if m_key in type_metrics:
                                    val = type_metrics[m_key]/cnt_type
                                    fmt_val = f"{val:.2f}" if val > 1 else f"{val:.4f}"
                                    metric_values.append(fmt_val)
                            pbar_log[deg_type] = f"({'/'.join(metric_values)})"
                    pbar.set_postfix(**pbar_log)

        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt

            # Add per-type metrics
            for deg_type, metrics in self.metric_results_per_type.items():
                cnt_type = metrics.pop('cnt', 0)
                if cnt_type > 0:
                    for m_name, m_val in metrics.items():
                        self.metric_results[f'{deg_type}/{m_name}'] = m_val / cnt_type

            self._log_validation_metric_values(
                current_iter, dataset_name, tb_logger, self.metric_results)
        return 0.

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

        metrics_dir = osp.join(self.opt['path']['log'], 'metric_results')
        os.makedirs(metrics_dir, exist_ok=True)

        safe_dataset_name = dataset_name.replace('/', '_').replace(' ', '_')
        safe_iter = str(current_iter).replace('/', '_').replace(' ', '_')
        payload = OrderedDict(
            dataset_name=dataset_name,
            current_iter=str(current_iter),
            metrics=OrderedDict(),
        )

        iter_path = osp.join(metrics_dir, f'{safe_dataset_name}_{safe_iter}.json')
        latest_path = osp.join(metrics_dir, f'{safe_dataset_name}_latest.json')

        for path in (iter_path, latest_path):
            if osp.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        old_payload = json.load(f)
                    old_metrics = old_payload.get('metrics', {})
                    if isinstance(old_metrics, dict):
                        for metric, value in old_metrics.items():
                            payload['metrics'][metric] = float(value)
                except Exception:
                    pass

        for metric, value in metric_dict.items():
            payload['metrics'][metric] = float(value)

        with open(iter_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def get_current_log(self):
        return self.log_dict
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters',
                                                  False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = self._torch_load_compat(
            load_path, map_location=lambda storage, loc: storage, weights_only=True)
        if param_key is not None:
            load_net = load_net[param_key]
        # print(' load net keys', load_net.keys())
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'],
                                     save_filename)
            torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
