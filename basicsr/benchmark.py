# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
import gc
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
import torch.nn as nn

try:
    from basicsr._bootstrap import setup_project_root
except ImportError:
    from _bootstrap import setup_project_root

setup_project_root(__file__)

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.models.ir_prior_model import IRPriorModel
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse
from basicsr.utils import check_resume, get_env_info, get_root_logger, get_time_str
from basicsr.utils import set_random_seed
from basicsr.utils.flops import set_extended_flop_handles


def parse_benchmark_options():
    """Parse benchmark-specific CLI flags and load the YAML options."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True,
                        help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--input_path', type=str, required=False,
                        help='Unused benchmark compatibility flag.')
    parser.add_argument('--output_path', type=str, required=False,
                        help='Unused benchmark compatibility flag.')
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--weights', type=str, default=None,
                        help='Override path.pretrain_network_g with a checkpoint path.')
    parser.add_argument('--phase', type=str, default=None,
                        help='Benchmark only one dataset phase, e.g. val or test.')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Warmup iterations before timing.')
    parser.add_argument('--iters', type=int, default=100,
                        help='Measured iterations for timing.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Optional override for benchmark dataloader batch size.')
    parser.add_argument('--skip_flops', action='store_true',
                        help='Skip fvcore FLOPs analysis.')
    parser.add_argument('--extended_flop_handles', action='store_true',
                        help='Enable project-added fvcore handlers for fused attention and elementwise ops.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=False)
    if args.gpu is not None:
        opt['gpu'] = args.gpu
    if 'gpu' in opt and opt['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['gpu'])

    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
        init_dist(args.launcher, **opt.get('dist_params', {})) if args.launcher == 'slurm' and 'dist_params' in opt else init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }
    if args.weights is not None:
        opt['path']['pretrain_network_g'] = osp.expanduser(args.weights)

    return opt, args


def setup_logger(opt):
    """Create benchmark log directory and logger without renaming old results."""
    os.makedirs(opt['path']['results_root'], exist_ok=True)
    os.makedirs(opt['path']['log'], exist_ok=True)
    log_file = osp.join(opt['path']['log'],
                        f"benchmark_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    return logger


def auto_resume_if_available(opt, logger):
    """Mimic test.py resume behavior and return loaded resume state if found."""
    state_folder_path = f'experiments/{opt["name"]}/training_states/'
    try:
        states = os.listdir(state_folder_path)
    except OSError:
        states = []

    resume_state = None
    if states:
        max_state_file = '{}.state'.format(max(int(x[0:-6]) for x in states))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state
        logger.info(f'Auto resume from {resume_state}')

    if opt['path'].get('resume_state'):
        if torch.cuda.is_available() and opt['num_gpu'] != 0:
            device_id = torch.cuda.current_device()
            return torch.load(
                opt['path']['resume_state'],
                map_location=lambda storage, loc: storage.cuda(device_id))
        return torch.load(opt['path']['resume_state'], map_location='cpu')
    return None


def build_model(opt, resume_state, logger):
    """Instantiate the configured model, following the same resume logic as test.py."""
    if resume_state:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        logger.info(f"Resuming benchmark target from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
    else:
        model = create_model(opt)
    return model


def create_benchmark_loaders(opt, args, logger):
    """Create benchmark dataloaders for the requested non-train phases."""
    loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase.startswith('train'):
            continue
        if args.phase and phase.split('_')[0] != args.phase:
            continue

        dataset_opt = deepcopy(dataset_opt)
        if phase.startswith('test'):
            dataset_opt['phase'] = 'test'
        phase_name = dataset_opt.get('phase', phase.split('_')[0])
        if args.batch_size is not None:
            dataset_opt['batch_size_per_gpu'] = args.batch_size

        dataset = create_dataset(dataset_opt)
        loader = create_dataloader(
            dataset,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f'Benchmark {phase_name} dataset {dataset_opt["name"]}: {len(dataset)} images, '
            f'batch_size={dataset_opt.get("batch_size_per_gpu", 1)}')
        loaders.append((phase, loader))

    if not loaders:
        raise ValueError('No non-train dataset found for benchmarking.')
    return loaders


def get_tensor_batch_size(inputs):
    """Infer batch size from the first tensor input."""
    for item in inputs:
        if torch.is_tensor(item):
            return int(item.shape[0]) if item.ndim > 0 else 1
    return 1


def count_parameters(module):
    """Return total and trainable parameter counts for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_count(num):
    """Pretty-print large counts using M/B units."""
    if num >= 1e9:
        return f'{num / 1e9:.4f} B'
    if num >= 1e6:
        return f'{num / 1e6:.4f} M'
    if num >= 1e3:
        return f'{num / 1e3:.4f} K'
    return str(num)


def format_flops(num):
    """Pretty-print FLOPs using G/M units."""
    if num is None:
        return 'N/A'
    if num >= 1e12:
        return f'{num / 1e12:.4f} TFLOPs'
    if num >= 1e9:
        return f'{num / 1e9:.4f} GFLOPs'
    if num >= 1e6:
        return f'{num / 1e6:.4f} MFLOPs'
    return f'{num:.4f} FLOPs'


def maybe_analyze_flops(module, inputs, skip_flops=False, extended_flop_handles=False):
    """Estimate FLOPs with fvcore, or return an explanatory message if unavailable."""
    if skip_flops:
        return None, 'skipped by --skip_flops'

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None, 'fvcore is not installed'

    try:
        with torch.no_grad():
            analysis = FlopCountAnalysis(module, inputs)
            if extended_flop_handles:
                analysis = set_extended_flop_handles(analysis)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            total_flops = analysis.total()
            unsupported = analysis.unsupported_ops()
        if unsupported:
            unsupported_str = ', '.join(f'{k}:{v}' for k, v in sorted(unsupported.items()))
            note = f'unsupported_ops={unsupported_str}'
        else:
            note = ''
        del analysis
        cleanup_cuda_memory(inputs[0].device if inputs and torch.is_tensor(inputs[0]) else torch.device('cpu'))
        return total_flops, note
    except Exception as exc:  # pragma: no cover - depends on external ops support
        cleanup_cuda_memory(inputs[0].device if inputs and torch.is_tensor(inputs[0]) else torch.device('cpu'))
        return None, f'fvcore failed: {exc}'


def synchronize_if_needed(device):
    """Synchronize the active CUDA stream before/after timing."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def cleanup_cuda_memory(device):
    """Release transient tensors between benchmark stages."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def benchmark_runtime(module, inputs, device, warmup=20, iters=100):
    """Measure throughput and per-image latency for one module."""
    module.eval()
    batch_size = get_tensor_batch_size(inputs)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            module(*inputs)
        synchronize_if_needed(device)
        cleanup_cuda_memory(device)

        start = time.perf_counter()
        for _ in range(max(iters, 1)):
            module(*inputs)
        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
    cleanup_cuda_memory(device)

    num_images = max(iters, 1) * batch_size
    images_per_second = num_images / elapsed if elapsed > 0 else float('inf')
    ms_per_image = elapsed * 1000.0 / num_images if num_images > 0 else float('inf')
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == 'cuda' else 0.0)
    return OrderedDict([
        ('elapsed_sec', elapsed),
        ('batch_size', batch_size),
        ('images_per_second', images_per_second),
        ('ms_per_image', ms_per_image),
        ('peak_memory_mb', peak_memory_mb),
    ])


class NetPriorRuntimeWrapper(nn.Module):
    """Benchmark net_prior with the same autocast behavior used in validation."""

    def __init__(self, net_prior, device):
        super().__init__()
        self.net_prior = net_prior
        self.device = device

    def forward(self, lq_clip):
        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                return self.net_prior(lq_clip)
        return self.net_prior(lq_clip)


class NetPriorFlopWrapper(nn.Module):
    """Benchmark net_prior FLOPs without relying on runtime autocast."""

    def __init__(self, net_prior):
        super().__init__()
        self.net_prior = net_prior

    def forward(self, lq_clip):
        param = next((p for p in self.net_prior.parameters() if p.is_floating_point()), None)
        if param is not None and lq_clip.is_floating_point() and lq_clip.dtype != param.dtype:
            lq_clip = lq_clip.to(dtype=param.dtype)
        return self.net_prior(lq_clip)


class EndToEndFlopWrapper(nn.Module):
    """Benchmark the full CoRE-UIR pipeline FLOPs without runtime autocast wrappers."""

    def __init__(self, net_g, mode, net_prior=None):
        super().__init__()
        self.net_g = net_g
        self.net_prior = net_prior
        self.mode = mode

    def forward(self, lq, lq_clip=None):
        if self.mode == 'image':
            return self.net_g(lq)

        prior_param = next((p for p in self.net_prior.parameters() if p.is_floating_point()), None)
        if prior_param is not None and lq_clip.is_floating_point() and lq_clip.dtype != prior_param.dtype:
            lq_clip = lq_clip.to(dtype=prior_param.dtype)
        prior_out = self.net_prior(lq_clip)

        if self.mode == 'ir2':
            prior = prior_out[1].float()
            return self.net_g(lq, prior)
        raise ValueError(f'Unsupported benchmark mode: {self.mode}')


class NetGWrapper(nn.Module):
    """Normalize net_g forward signatures across Image/IR2 wrappers."""

    def __init__(self, net_g, mode):
        super().__init__()
        self.net_g = net_g
        self.mode = mode

    def forward(self, lq, prior=None, probs=None):
        if self.mode == 'ir2':
            return self.net_g(lq, prior)
        return self.net_g(lq)


class EndToEndWrapper(nn.Module):
    """Benchmark the full inference path using the model's actual submodules."""

    def __init__(self, net_g, mode, device, net_prior=None):
        super().__init__()
        self.net_g = net_g
        self.net_prior = net_prior
        self.mode = mode
        self.device = device

    def forward(self, lq, lq_clip=None):
        if self.mode == 'image':
            return self.net_g(lq)

        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                prior_out = self.net_prior(lq_clip)
        else:
            prior_out = self.net_prior(lq_clip)

        if self.mode == 'ir2':
            prior = prior_out[1].float()
            return self.net_g(lq, prior)
        raise ValueError(f'Unsupported benchmark mode: {self.mode}')


def prepare_component_inputs(model, batch):
    """Prepare CPU-side benchmark inputs and model mode metadata."""
    if isinstance(model, IRPriorModel) or hasattr(model, 'net_prior'):
        mode = 'ir2'
    elif isinstance(model, ImageRestorationModel):
        mode = 'image'
    else:
        mode = 'image'

    payload = {
        'mode': mode,
        'lq_cpu': batch['lq'].cpu(),
        'lq_clip_cpu': batch.get('lq_clip').cpu() if batch.get('lq_clip') is not None else None,
    }
    return payload


def move_batch_to_device(payload, device):
    """Move benchmark batch tensors to the target device on demand."""
    lq = payload['lq_cpu'].to(device)
    lq_clip = payload['lq_clip_cpu'].to(device) if payload['lq_clip_cpu'] is not None else None
    return lq, lq_clip


def build_net_g_inputs(model, mode, lq, lq_clip, device):
    """Construct realistic net_g inputs using the model's router output when needed."""
    if mode == 'image':
        return (lq,)

    prior_runtime = NetPriorRuntimeWrapper(model.get_bare_model(model.net_prior), device)
    prior_runtime.eval()
    with torch.no_grad():
        prior_out = prior_runtime(lq_clip)

    if mode == 'ir2':
        prior = prior_out[1].float()
        return (lq, prior)
    raise ValueError(f'Unsupported benchmark mode: {mode}')


def slice_inputs_for_flops(inputs):
    """Use batch size 1 for FLOPs analysis to reduce tracing memory."""
    sliced = []
    for item in inputs:
        if torch.is_tensor(item) and item.ndim > 0:
            sliced.append(item[:1].contiguous())
        else:
            sliced.append(item)
    return tuple(sliced)


def log_component_stats(logger, title, params_total, params_trainable, flops, flop_note, runtime):
    """Emit a compact benchmark summary for one component."""
    logger.info(f'[{title}]')
    logger.info(f'  params_total: {format_count(params_total)} ({params_total})')
    logger.info(f'  params_trainable: {format_count(params_trainable)} ({params_trainable})')
    logger.info(f'  flops: {format_flops(flops)}')
    if flop_note:
        logger.info(f'  flops_note: {flop_note}')
    logger.info(f'  batch_size: {runtime["batch_size"]}')
    logger.info(f'  images_per_second: {runtime["images_per_second"]:.4f}')
    logger.info(f'  ms_per_image: {runtime["ms_per_image"]:.4f}')
    logger.info(f'  peak_memory_mb: {runtime["peak_memory_mb"]:.2f}')


def benchmark_one_loader(model, loader, args, logger):
    """Benchmark the configured model on the first batch of one evaluation loader."""
    batch = next(iter(loader))
    payload = prepare_component_inputs(model, batch)
    device = model.device

    logger.info(f'Using sample batch from dataset {loader.dataset.opt["name"]}')
    logger.info(f'  lq shape: {tuple(payload["lq_cpu"].shape)}')
    if payload['lq_clip_cpu'] is not None:
        logger.info(f'  lq_clip shape: {tuple(payload["lq_clip_cpu"].shape)}')

    bare_net_g = model.get_bare_model(model.net_g)
    bare_net_prior = model.get_bare_model(model.net_prior) if hasattr(model, 'net_prior') else None
    net_g_wrapper = NetGWrapper(bare_net_g, payload['mode']).to(device)
    end_to_end_runtime = EndToEndWrapper(
        bare_net_g, payload['mode'], device, net_prior=bare_net_prior).to(device)
    end_to_end_flop = EndToEndFlopWrapper(
        bare_net_g, payload['mode'], net_prior=bare_net_prior).to(device)

    lq, lq_clip = move_batch_to_device(payload, device)
    net_g_inputs = build_net_g_inputs(model, payload['mode'], lq, lq_clip, device)
    net_g_flop_inputs = slice_inputs_for_flops(net_g_inputs)

    net_g_params = count_parameters(bare_net_g)
    net_prior_params = (0, 0)
    net_g_flops, net_g_flop_note = maybe_analyze_flops(
        net_g_wrapper, net_g_flop_inputs,
        skip_flops=args.skip_flops,
        extended_flop_handles=args.extended_flop_handles)
    net_g_runtime = benchmark_runtime(
        net_g_wrapper, net_g_inputs, device,
        warmup=args.warmup, iters=args.iters)
    log_component_stats(
        logger, 'net_g', net_g_params[0], net_g_params[1],
        net_g_flops, net_g_flop_note, net_g_runtime)
    del net_g_inputs, net_g_flop_inputs
    cleanup_cuda_memory(device)

    if payload['lq_clip_cpu'] is not None and hasattr(model, 'net_prior'):
        prior_runtime_wrapper = NetPriorRuntimeWrapper(bare_net_prior, device).to(device)
        prior_flop_wrapper = NetPriorFlopWrapper(bare_net_prior).to(device)
        net_prior_params = count_parameters(bare_net_prior)
        prior_flop_inputs = slice_inputs_for_flops((lq_clip,))
        net_prior_flops, net_prior_flop_note = maybe_analyze_flops(
            prior_flop_wrapper, prior_flop_inputs,
            skip_flops=args.skip_flops,
            extended_flop_handles=args.extended_flop_handles)
        net_prior_runtime = benchmark_runtime(
            prior_runtime_wrapper, (lq_clip,), device,
            warmup=args.warmup, iters=args.iters)
        log_component_stats(
            logger, 'net_prior', net_prior_params[0], net_prior_params[1],
            net_prior_flops, net_prior_flop_note, net_prior_runtime)
        del prior_flop_inputs
        cleanup_cuda_memory(device)
    else:
        logger.info('[net_prior]')
        logger.info('  not present in this model wrapper')

    if payload['mode'] == 'image':
        end_to_end_inputs = (lq,)
    else:
        end_to_end_inputs = (lq, lq_clip)
    end_to_end_flop_inputs = slice_inputs_for_flops(end_to_end_inputs)

    e2e_flops, e2e_flop_note = maybe_analyze_flops(
        end_to_end_flop, end_to_end_flop_inputs,
        skip_flops=args.skip_flops,
        extended_flop_handles=args.extended_flop_handles)
    e2e_runtime = benchmark_runtime(
        end_to_end_runtime, end_to_end_inputs, device,
        warmup=args.warmup, iters=args.iters)
    logger.info('[end_to_end]')
    e2e_params_total = net_g_params[0] + net_prior_params[0]
    e2e_params_trainable = net_g_params[1] + net_prior_params[1]
    logger.info(f'  params_total: {format_count(e2e_params_total)} ({e2e_params_total})')
    logger.info(f'  params_trainable: {format_count(e2e_params_trainable)} ({e2e_params_trainable})')
    logger.info(f'  flops: {format_flops(e2e_flops)}')
    if e2e_flop_note:
        logger.info(f'  flops_note: {e2e_flop_note}')
    logger.info(f'  batch_size: {e2e_runtime["batch_size"]}')
    logger.info(f'  images_per_second: {e2e_runtime["images_per_second"]:.4f}')
    logger.info(f'  ms_per_image: {e2e_runtime["ms_per_image"]:.4f}')
    logger.info(f'  peak_memory_mb: {e2e_runtime["peak_memory_mb"]:.2f}')
    del lq, lq_clip, end_to_end_inputs, end_to_end_flop_inputs
    cleanup_cuda_memory(device)


def main():
    opt, args = parse_benchmark_options()
    torch.backends.cudnn.benchmark = True

    logger = setup_logger(opt)
    resume_state = None if args.weights is not None else auto_resume_if_available(opt, logger)
    model = build_model(opt, resume_state, logger)
    loaders = create_benchmark_loaders(opt, args, logger)

    logger.info(f'Benchmark warmup={args.warmup}, iters={args.iters}')
    if args.skip_flops:
        logger.info('FLOPs analysis disabled by --skip_flops')
    elif args.extended_flop_handles:
        logger.info('FLOPs analysis uses project-added extended op handlers')
    else:
        logger.info('FLOPs analysis uses fvcore default op handlers')

    for phase, loader in loaders:
        logger.info('=' * 80)
        logger.info(f'Benchmarking phase={phase}, dataset={loader.dataset.opt["name"]}')
        benchmark_one_loader(model, loader, args, logger)


if __name__ == '__main__':
    main()
