# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging
import torch
import os
from os import path as osp
try:
    from basicsr._bootstrap import setup_project_root
except ImportError:
    from _bootstrap import setup_project_root

setup_project_root(__file__)

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options, find_latest_resume_state
from basicsr.utils import (get_env_info, get_root_logger, get_time_str, check_resume,
                           make_exp_dirs)
from basicsr.utils.options import dict2str


def main():
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(is_train=False)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    resume_state = None
    if args.weights is None and not args.no_resume:
        resume_state = find_latest_resume_state(opt)
    if resume_state is not None:
        opt['path']['resume_state'] = resume_state
    else:
        opt['path']['resume_state'] = None

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # create model
    if resume_state:  # resume training
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        # model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
    else:
        model = create_model(opt)

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase.startswith('train'):
            continue  # skip training phase
        if phase.startswith('test'):
            dataset_opt['phase'] = 'test'
        phase_name = dataset_opt.get('phase', phase.split('_')[0])
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of {phase_name} images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append((phase_name, test_loader))

    for phase_name, test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Running {phase_name} split {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            # save_img=opt['val']['save_img'],
            save_img=True,
            rgb2bgr=rgb2bgr, use_image=use_image)


if __name__ == '__main__':
    main()
