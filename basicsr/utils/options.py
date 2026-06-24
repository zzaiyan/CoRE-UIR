# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import yaml
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import re


def _deep_merge_dict(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _dataset_collection_items(dataset_opt):
    if not isinstance(dataset_opt, dict):
        return None
    datasets = dataset_opt.get('datasets')
    if datasets is None:
        return None
    if not isinstance(datasets, list):
        raise TypeError('datasets.<split>.datasets must be a list of dataset option dicts.')
    if not datasets:
        raise ValueError('datasets.<split>.datasets must contain at least one dataset option dict.')
    return datasets


def _dataset_key_suffix(dataset_opt, index):
    value = dataset_opt.get('name') or dataset_opt.get('type') or f'dataset_{index + 1}'
    suffix = re.sub(r'[^0-9A-Za-z]+', '_', str(value)).strip('_').lower()
    return suffix or f'dataset_{index + 1}'


def _unique_dataset_key(split_key, dataset_opt, index, existing_keys):
    suffix = _dataset_key_suffix(dataset_opt, index)
    candidate = f'{split_key}_{suffix}'
    counter = 2
    while candidate in existing_keys:
        candidate = f'{split_key}_{suffix}_{counter}'
        counter += 1
    return candidate


def _normalize_multi_dataset_splits(opt):
    datasets_opt = opt.get('datasets')
    if not isinstance(datasets_opt, dict):
        return

    normalized = OrderedDict()
    for split_key, dataset_opt in datasets_opt.items():
        child_datasets = _dataset_collection_items(dataset_opt)
        if child_datasets is None:
            normalized[split_key] = dataset_opt
            continue

        shared_opt = OrderedDict((key, deepcopy(value)) for key, value in dataset_opt.items() if key != 'datasets')
        phase = str(split_key).split('_')[0]
        if phase == 'train':
            composite_opt = deepcopy(shared_opt)
            composite_opt.setdefault('name', str(split_key))
            composite_opt['datasets'] = []
            child_keys = set()
            for index, child_opt in enumerate(child_datasets):
                if not isinstance(child_opt, dict):
                    raise TypeError('Each train sub-dataset option must be a dict.')
                merged = _deep_merge_dict(shared_opt, child_opt)
                merged.pop('datasets', None)
                if 'name' not in child_opt and len(child_datasets) > 1:
                    merged['name'] = f'{composite_opt["name"]}_{index + 1}'
                child_split_key = _unique_dataset_key(split_key, merged, index, child_keys)
                child_keys.add(child_split_key)
                merged['_child_split_key'] = child_split_key
                composite_opt['datasets'].append(merged)
            normalized[split_key] = composite_opt
            continue

        for index, child_opt in enumerate(child_datasets):
            if not isinstance(child_opt, dict):
                raise TypeError(f'Each {phase} sub-dataset option must be a dict.')
            merged = _deep_merge_dict(shared_opt, child_opt)
            merged.pop('datasets', None)
            if 'name' not in child_opt and 'name' in shared_opt and len(child_datasets) > 1:
                merged['name'] = f'{shared_opt["name"]}_{index + 1}'
            child_key = _unique_dataset_key(split_key, merged, index, normalized)
            normalized[child_key] = merged

    opt['datasets'] = normalized


def _apply_dataset_runtime_meta(dataset_opt, split_key, scale=None):
    phase = str(split_key).split('_')[0]
    dataset_opt['split_key'] = split_key
    dataset_opt['phase'] = phase
    if scale is not None:
        dataset_opt['scale'] = scale

    child_datasets = _dataset_collection_items(dataset_opt)
    if child_datasets is None:
        return

    for index, child_opt in enumerate(child_datasets):
        child_split_key = child_opt.pop('_child_split_key', None)
        if child_split_key is None:
            child_split_key = _unique_dataset_key(split_key, child_opt, index, set())
        child_opt['split_key'] = child_split_key
        child_opt['phase'] = phase
        if scale is not None:
            child_opt['scale'] = scale


def normalize_datasets_options(datasets_opt, scale=None):
    if not isinstance(datasets_opt, dict):
        raise TypeError('datasets must be a mapping of split names to dataset option dicts.')

    opt = {'datasets': deepcopy(datasets_opt)}
    _normalize_multi_dataset_splits(opt)
    for split_key, dataset_opt in opt['datasets'].items():
        _apply_dataset_runtime_meta(dataset_opt, split_key, scale)
    return opt['datasets']


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # datasets
    if 'datasets' in opt:
        opt['datasets'] = normalize_datasets_options(opt['datasets'], opt.get('scale'))

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = './'
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['tb_logger'] = osp.join(experiments_root, 'tb_logs')
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['tb_logger'] = osp.join(results_root, 'tb_logs')
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
