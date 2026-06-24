"""Helpers for YAML-backed prior training configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from basicsr.utils.options import normalize_datasets_options

try:
    from prior._bootstrap import setup_project_root
except ImportError:
    from _bootstrap import setup_project_root

ROOT_DIR = setup_project_root(__file__)
PRIOR_DIR = ROOT_DIR / 'prior'
PRIOR_CONFIG_DIR = PRIOR_DIR / 'config'


def resolve_prior_config_path(config_path):
    """Resolve a config path against common project-relative locations."""
    if config_path is None:
        return None

    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        ROOT_DIR / path,
        PRIOR_CONFIG_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def load_prior_config(config_path):
    """Load a prior YAML config as a flat defaults dictionary."""
    resolved_path = resolve_prior_config_path(config_path)
    if resolved_path is None:
        return {}, None
    if not resolved_path.exists():
        raise FileNotFoundError(f'Config file not found: {resolved_path}')

    with open(resolved_path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f'Config file must contain a mapping at the top level: {resolved_path}')
    return data, resolved_path


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(',') if item.strip()]


def _dataset_collection_items(dataset_opt):
    if not isinstance(dataset_opt, dict):
        return None
    datasets = dataset_opt.get('datasets')
    if isinstance(datasets, list):
        return datasets
    return None


def normalize_prior_datasets_config(config_data):
    datasets = config_data.get('datasets')
    if not isinstance(datasets, dict):
        raise ValueError('Prior config must define a top-level datasets mapping.')
    return normalize_datasets_options(datasets)


def get_prior_phase_dataset_entries(datasets, phase):
    phase = str(phase)
    entries = []
    for split_key, dataset_opt in datasets.items():
        if split_key == phase or str(split_key).startswith(f'{phase}_'):
            entries.append((split_key, dataset_opt))
    return entries


def resolve_prior_dataset_entry(datasets, split):
    if split in datasets:
        return split, datasets[split]

    matches = get_prior_phase_dataset_entries(datasets, split)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f'Config split={split} is ambiguous because multiple datasets match that phase. '
            'Please use the expanded split key instead.')
    raise KeyError(f'Dataset split not found: {split}')


def infer_prior_degrade_types(datasets):
    def _extract(dataset_opt):
        children = _dataset_collection_items(dataset_opt)
        if children is not None:
            extracted = [_extract(child_opt) for child_opt in children]
            baseline = extracted[0]
            for current in extracted[1:]:
                if current != baseline:
                    raise ValueError('All prior dataset splits must share the same degrade_type ordering.')
            return baseline

        degrade_types = _to_list(dataset_opt.get('degrade_type'))
        if not degrade_types:
            raise ValueError('Each prior dataset config must define a non-empty degrade_type list.')
        return tuple(degrade_types)

    candidate_entries = get_prior_phase_dataset_entries(datasets, 'train') or list(datasets.items())
    baseline = None
    for _, dataset_opt in candidate_entries:
        current = _extract(dataset_opt)
        if baseline is None:
            baseline = current
            continue
        if current != baseline:
            raise ValueError('All prior dataset splits must share the same degrade_type ordering.')
    if baseline is None:
        raise ValueError('Prior config must define at least one dataset split.')
    return list(baseline)


def infer_config_name(config_path, explicit_name=None, fallback_name=None):
    """Infer a stable experiment name from YAML path or explicit override."""
    if explicit_name:
        return str(explicit_name)
    if config_path:
        return Path(config_path).stem
    if fallback_name:
        return str(fallback_name)
    return 'default'


def parse_args_with_yaml(build_parser_fn, default_config=None, fallback_name=None):
    """Parse CLI args with YAML defaults while preserving CLI precedence."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str)
    pre_parser.add_argument('--config_name', type=str)
    pre_args, _ = pre_parser.parse_known_args()

    config_arg = pre_args.config or default_config
    defaults, resolved_config = load_prior_config(config_arg)
    defaults = deepcopy(defaults)

    parser = build_parser_fn(defaults)
    parser.set_defaults(
        config=str(resolved_config) if resolved_config else None,
        config_name=infer_config_name(
            resolved_config,
            explicit_name=pre_args.config_name or defaults.get('config_name'),
            fallback_name=fallback_name or defaults.get('dataset_name')))
    args = parser.parse_args()

    if args.config:
        args.config = str(resolve_prior_config_path(args.config))
    args.config_name = infer_config_name(
        args.config,
        explicit_name=args.config_name,
        fallback_name=fallback_name or getattr(args, 'dataset_name', None))
    args.raw_config = deepcopy(defaults)
    args.datasets = normalize_prior_datasets_config(defaults)
    return args
