# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import sys
try:
    from basicsr._bootstrap import setup_project_root
except ImportError:
    from _bootstrap import setup_project_root

setup_project_root(__file__)

from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.models import create_model
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data import create_dataloader, create_dataset
import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
from os import path as osp
from tqdm import tqdm


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False,
                        help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False,
                        help='The path to the output image. For single image inference only.')
    parser.add_argument(
        '--weights',
        type=str,
        required=False,
        help='Override path.pretrain_network_g with a checkpoint path.')
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Skip automatic resume from the latest training state.')
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()

    opt = parse(args.opt, is_train=is_train)
    if args.gpu is not None:
        opt['gpu'] = args.gpu
    if 'gpu' in opt and opt['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['gpu'])

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
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


def find_latest_resume_state(opt):
    """Return the newest training-state path for the current experiment.

    The helper is shared by `train.py` and `test.py` so CLI flags can cleanly
    opt out of CoRE-UIR's auto-resume behavior without duplicating path logic.
    """
    experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
    state_folder_path = opt['path'].get('training_states')
    if not state_folder_path:
        state_folder_path = osp.join(experiments_root, 'training_states')
    if not osp.isdir(state_folder_path):
        return None

    state_files = []
    for filename in os.listdir(state_folder_path):
        if filename.endswith('.state') and filename[:-6].isdigit():
            state_files.append(filename)
    if not state_files:
        return None

    max_iter = max(int(filename[:-6]) for filename in state_files)
    return osp.join(state_folder_path, f'{max_iter}.state')


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=opt['path']['tb_logger'])
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    total_epochs, total_iters = None, None
    train_sampler = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase.startswith('val'):
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
            val_loaders.append(val_loader)
        elif phase.startswith('test'):
            logger.info(
                f'Skip auxiliary dataset phase {phase} during training. '
                'Use basicsr/test.py to run export/evaluation on this split.')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    resume_state = None
    if not args.no_resume:
        resume_state = find_latest_resume_state(opt)
    if resume_state is not None:
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger. For cross-experiment resume, the target
    # experiment directory may not exist yet, but we still need log/model dirs.
    if resume_state is None:
        make_exp_dirs(opt)
    else:
        for key in ('experiments_root', 'models', 'training_states', 'log',
                    'tb_logger'):
            path = opt['path'].get(key)
            if path is not None:
                os.makedirs(path, exist_ok=True)

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # Calculate model parameters statistics
    total_params = sum(p.numel() for p in model.net_g.parameters())
    trainable_params = sum(p.numel()
                           for p in model.net_g.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params * 100 if total_params > 0 else 0

    logger.info(f"Model parameter statistics:")
    logger.info(f"  Total parameters: {total_params/1e6:.4f} M")
    logger.info(f"  Trainable parameters: {trainable_params/1e6:.4f} M")
    logger.info(f"  Trainable ratio: {trainable_ratio:.4f} %")

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    def run_validation_loaders(current_iter):
        if opt.get('val') is None or not val_loaders:
            return None

        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        last_metric = None
        for val_loader in val_loaders:
            last_metric = model.validation(
                val_loader,
                current_iter=current_iter,
                tb_logger=tb_logger,
                save_img=opt['val']['save_img'],
                rgb2bgr=rgb2bgr,
                use_image=use_image)
        return last_metric

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # Create progress bar for total iterations
    pbar = tqdm(total=total_iters, initial=current_iter,
                desc="Training", unit="iter")

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            result_code = model.optimize_parameters(current_iter, tb_logger)
            # if result_code == -1 and tb_logger:
            #     print('loss explode .. ')
            #     exit(0)
            iter_time = time.time() - iter_time

            # Update progress bar
            pbar.update(1)
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter,
                            'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                # print('msg logger .. ', current_iter)
                msg_logger(log_vars)

                # Update progress bar description with current loss info
                current_log = model.get_current_log()
                loss_str = ""
                if 'l_total' in current_log:
                    loss_str = f"Loss: {current_log['l_total']:.4f}"
                elif 'loss' in current_log:
                    loss_str = f"Loss: {current_log['loss']:.4f}"
                else:
                    for key, value in current_log.items():
                        if key.startswith('l_'):
                            loss_str = f"{key}: {value:.4f}"
                            break
                pbar.set_description(f"Epoch {epoch}, {loss_str}")

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                # logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0 or current_iter == 1000):
                # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                run_validation_loaders(current_iter)
                log_vars = {'epoch': epoch, 'iter': current_iter,
                            'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                pbar.set_postfix(**{k: f'{v:.4f}' for k, v in model.log_dict.items()})

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    # Close progress bar
    pbar.close()

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None and val_loaders:
        metric = run_validation_loaders(current_iter)
        # if tb_logger:
        #     print('xxresult! ', opt['name'], ' ', metric)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
    main()
