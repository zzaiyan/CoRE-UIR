import argparse
import os
from copy import deepcopy
from os import path as osp
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

try:
    from prior._bootstrap import setup_project_root
except ImportError:
    from _bootstrap import setup_project_root

ROOT_DIR = setup_project_root(__file__)
PRIOR_DIR = ROOT_DIR / 'prior'
DEFAULT_CKPT_DIR = PRIOR_DIR / 'checkpoints'

from basicsr.data import create_dataset  # noqa: E402
from prior.config_utils import (  # noqa: E402
    get_prior_phase_dataset_entries,
    infer_prior_degrade_types,
    parse_args_with_yaml,
)
from prior.classification_utils import (  # noqa: E402
    build_classification_loss,
    build_classification_spec,
    summarize_prediction_outputs,
)
from prior.model import (  # noqa: E402
    build_prior_router,
    normalize_prior_router_state_dict,
    router_default_view_mode,
    router_checkpoint_stem,
)
from prior.utils_image import VALID_VIEW_MODES  # noqa: E402


def config_checkpoint_dir(ckpt_root, config_name):
    """Return the checkpoint directory for one named prior experiment."""
    return Path(ckpt_root) / str(config_name)


def dataset_checkpoint_dir(ckpt_root, dataset_name):
    """Backward-compatible alias for older imports."""
    return config_checkpoint_dir(ckpt_root, dataset_name)


def parse_csv_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(',') if item.strip()]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def prepare_single_gpu_context(gpu):
    """Switch the default CUDA context to the user-selected GPU early.

    Lightning may probe CUDA before it moves the model, which can create a
    small context allocation on ``cuda:0`` if the current device is left at
    the PyTorch default. Setting the current device up front keeps those
    framework-side allocations on the intended GPU.
    """
    if not torch.cuda.is_available():
        return

    gpu = int(gpu)
    device_count = torch.cuda.device_count()
    if gpu < 0 or gpu >= device_count:
        raise ValueError(
            f'Invalid --gpu={gpu}. Available CUDA devices: 0..{device_count - 1}.')
    torch.cuda.set_device(gpu)


def build_router_model(
        router_type,
        cls_num,
        *,
        router_dim=384,
        view_mode=None,
        view_size=None,
        clip_vision_tower='openai/clip-vit-base-patch32'):
    return build_prior_router(
        router_type=router_type,
        cls_num=cls_num,
        dim=router_dim,
        view_mode=view_mode,
        view_size=view_size,
        vision_tower=clip_vision_tower)


def resolve_router_view_args(args, router_type=None, prefix=''):
    router_type = router_type or args.router_type
    view_mode = getattr(args, f'{prefix}view_mode', None)
    view_size = getattr(args, f'{prefix}view_size', None)

    if view_mode is None:
        view_mode = router_default_view_mode(router_type)
    if view_size is None:
        view_size = int(args.patch_size)
    return str(view_mode), int(view_size)


def build_router_model_from_args(args, cls_num, router_type=None):
    view_mode, view_size = resolve_router_view_args(args, router_type=router_type)
    return build_router_model(
        router_type=router_type or args.router_type,
        cls_num=cls_num,
        router_dim=args.router_dim,
        view_mode=view_mode,
        view_size=view_size,
        clip_vision_tower=args.clip_vision_tower)


def build_dataset_opt(args, dataset_opt, router_type=None):
    dataset_opt = deepcopy(dataset_opt)
    router_type = router_type or args.router_type
    view_mode, view_size = resolve_router_view_args(args, router_type=router_type)

    def _apply_runtime_defaults(current_opt):
        phase = current_opt.get('phase', 'train')
        current_opt.setdefault('name', current_opt.get('split_key', f'{args.dataset_name}-{phase}'))
        current_opt.setdefault('view_mode', view_mode)
        current_opt.setdefault('view_size', view_size)
        current_opt.setdefault('clip_transform_mode', 'train' if phase == 'train' else 'inference')
        current_opt.setdefault('gt_size', args.crop_size)
        current_opt.setdefault('edge_decay', 0.0)
        current_opt.setdefault('data_augment', phase == 'train')
        current_opt.setdefault('cache_memory', bool(args.cache_memory) if phase == 'train' else False)

        child_datasets = current_opt.get('datasets') if isinstance(current_opt, dict) else None
        if isinstance(child_datasets, list):
            current_opt['datasets'] = [
                _apply_runtime_defaults(deepcopy(child_opt))
                for child_opt in child_datasets
            ]
        return current_opt

    return _apply_runtime_defaults(dataset_opt)


class DPEModule(pl.LightningModule):
    def __init__(self, args, degrade_types):
        super().__init__()
        self.degrade_types = list(degrade_types)
        self.classification_spec = build_classification_spec(
            self.degrade_types,
            multi_label=bool(args.multi_label_classification),
            dataset_name=args.dataset_name,
            threshold=args.multi_label_threshold)
        self.label_names = list(self.classification_spec.label_names)
        self.save_hyperparameters({
            'degrade_types': self.degrade_types,
            'classification_labels': self.label_names,
            'multi_label_classification': self.classification_spec.multi_label,
            'router_type': args.router_type,
            'lr': args.lr,
            'epochs': args.epochs
        })
        self.model = build_router_model_from_args(
            args, cls_num=self.classification_spec.cls_num)
        self.criterion = build_classification_loss(
            multi_label=self.classification_spec.multi_label,
            label_smoothing=args.label_smoothing)
        self.lr = args.lr
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.adam_betas = (args.adam_beta1, args.adam_beta2)
        self.eta_min = args.eta_min
        self.validation_outputs = {}
        self.test_outputs = {}

    def forward(self, x):
        return self.model(x)

    def _encode_labels(self, degrade_types, device):
        return self.classification_spec.encode_batch(degrade_types, device=device)

    def _stage_dataloaders(self, stage):
        if self.trainer is None:
            return []
        dataloaders = self.trainer.val_dataloaders if stage == 'val' else self.trainer.test_dataloaders
        if dataloaders is None:
            return []
        if isinstance(dataloaders, (list, tuple)):
            return list(dataloaders)
        return [dataloaders]

    def _dataset_name_for_loader(self, stage, dataloader_idx):
        dataloaders = self._stage_dataloaders(stage)
        if dataloader_idx < len(dataloaders):
            dataset = dataloaders[dataloader_idx].dataset
            return dataset.opt.get('name', dataset.opt.get('split_key', f'{stage}_{dataloader_idx}'))
        return f'{stage}_{dataloader_idx}'

    def _safe_metric_name(self, value):
        return str(value).replace('/', '_').replace(' ', '_')

    def _shared_eval_step(self, batch, stage, dataloader_idx=0):
        images = batch['lq_clip']
        labels = self._encode_labels(batch['degrade_type'], images.device)
        dataset_name = self._dataset_name_for_loader(stage, dataloader_idx)
        safe_dataset_name = self._safe_metric_name(dataset_name)

        outputs, _ = self(images)
        loss = self.criterion(outputs, labels)
        preds = self.classification_spec.predict_from_logits(outputs)

        item = {
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu(),
            'degrade_type': list(batch['degrade_type']),
            'loss': float(loss.detach().cpu()),
            'batch_size': labels.size(0)
        }

        if stage == 'val':
            self.validation_outputs.setdefault(dataloader_idx, []).append(item)
            self.log(
                f'val_loss/{safe_dataset_name}',
                loss,
                prog_bar=False,
                batch_size=labels.size(0))
        else:
            self.test_outputs.setdefault(dataloader_idx, []).append(item)
            self.log(
                f'test_loss/{safe_dataset_name}',
                loss,
                prog_bar=False,
                batch_size=labels.size(0))

        return loss

    def _log_eval_summary(self, stage, outputs, dataset_name, dataloader_idx):
        summary = summarize_prediction_outputs(outputs, self.classification_spec)
        if summary is None:
            return

        safe_dataset_name = self._safe_metric_name(dataset_name)

        title = 'Validation' if stage == 'val' else 'Test'
        self.print('\n' + '=' * 60)
        self.print(f'{title} Results [{dataset_name}]:')

        bucket_title = 'Sample-Type Exact Match' if self.classification_spec.multi_label else 'Per-Type Accuracy'
        self.print(f'  {bucket_title}:')
        for item in summary['per_type_accuracy']:
            degrade_type = item['degrade_type']
            correct = item['correct']
            total = item['total']
            acc = item['accuracy'] if item['total'] > 0 else 0.0
            self.print(
                f'  {degrade_type:12s}: {correct:5.0f}/{total:5.0f} = {acc:.2%}')
            self.log(
                f'{stage}_acc/{safe_dataset_name}/{degrade_type}',
                acc,
                prog_bar=False)

        overall_acc = float(summary['overall_accuracy'])
        if self.classification_spec.multi_label:
            self.print(f'  {"ExactMatch":12s}: {overall_acc:.2%}')
            self.print(f'  {"Hamming":12s}: {float(summary["hamming_accuracy"]):.2%}')
            self.print(f'  {"MicroF1":12s}: {float(summary["micro_f1"]):.2%}')
            self.print(f'  {"MacroF1":12s}: {float(summary["macro_f1"]):.2%}')
            self.print('  Per-Label Metrics:')
            for item in summary['per_label_metrics']:
                label_name = item['label']
                self.print(
                    f'  {label_name:12s}: '
                    f'P={float(item["precision"]):.2%} '
                    f'R={float(item["recall"]):.2%} '
                    f'F1={float(item["f1"]):.2%} '
                    f'Acc={float(item["accuracy"]):.2%}')
                self.log(
                    f'{stage}_precision/{safe_dataset_name}/{label_name}',
                    float(item['precision']),
                    prog_bar=False)
                self.log(
                    f'{stage}_recall/{safe_dataset_name}/{label_name}',
                    float(item['recall']),
                    prog_bar=False)
                self.log(
                    f'{stage}_f1/{safe_dataset_name}/{label_name}',
                    float(item['f1']),
                    prog_bar=False)
            self.log(
                f'{stage}_acc_hamming/{safe_dataset_name}',
                float(summary['hamming_accuracy']),
                prog_bar=False)
            self.log(
                f'{stage}_f1_micro/{safe_dataset_name}',
                float(summary['micro_f1']),
                prog_bar=stage == 'val' and dataloader_idx == 0)
            self.log(
                f'{stage}_f1_macro/{safe_dataset_name}',
                float(summary['macro_f1']),
                prog_bar=False)
        else:
            total_correct = sum(int(item['correct']) for item in summary['per_type_accuracy'])
            total_samples = sum(int(item['total']) for item in summary['per_type_accuracy'])
            self.print(
                f'  {"Overall":12s}: {total_correct:5.0f}/{total_samples:5.0f} = {overall_acc:.2%}')
        self.print('=' * 60 + '\n')
        self.log(
            f'{stage}_acc_overall/{safe_dataset_name}',
            overall_acc,
            prog_bar=stage == 'val' and dataloader_idx == 0)

    def training_step(self, batch, batch_idx):
        images = batch['lq_clip']
        labels = self._encode_labels(batch['degrade_type'], images.device)

        outputs, _ = self(images)
        loss = self.criterion(outputs, labels)
        preds = self.classification_spec.predict_from_logits(outputs)
        acc = self.classification_spec.batch_accuracy(preds, labels)

        self.log('train_loss', loss, prog_bar=True, batch_size=labels.size(0))
        self.log('train_acc', acc, prog_bar=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, 'val', dataloader_idx=dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, 'test', dataloader_idx=dataloader_idx)

    def on_validation_epoch_start(self):
        self.validation_outputs = {}

    def on_test_epoch_start(self):
        self.test_outputs = {}

    def on_validation_epoch_end(self):
        for dataloader_idx in sorted(self.validation_outputs):
            self._log_eval_summary(
                'val',
                self.validation_outputs[dataloader_idx],
                self._dataset_name_for_loader('val', dataloader_idx),
                dataloader_idx)

    def on_test_epoch_end(self):
        for dataloader_idx in sorted(self.test_outputs):
            self._log_eval_summary(
                'test',
                self.test_outputs[dataloader_idx],
                self._dataset_name_for_loader('test', dataloader_idx),
                dataloader_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.eta_min)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


class AllInOneClassificationDataModule(pl.LightningDataModule):
    def __init__(self, args, degrade_types):
        super().__init__()
        self.args = args
        self.degrade_types = degrade_types
        self.train_dataset = None
        self.val_datasets = []
        self.test_datasets = []

    def _build_phase_datasets(self, phase, *, fallback_phase=None):
        entries = get_prior_phase_dataset_entries(self.args.datasets, phase)
        if not entries and fallback_phase is not None:
            entries = get_prior_phase_dataset_entries(self.args.datasets, fallback_phase)
        datasets = []
        for _, raw_opt in entries:
            dataset_opt = build_dataset_opt(self.args, raw_opt)
            dataset = create_dataset(dataset_opt)
            datasets.append(dataset)
        return datasets

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            train_raw_opt = self.args.datasets.get('train')
            if train_raw_opt is None:
                raise ValueError('Prior config must define datasets.train for router training.')
            train_opt = build_dataset_opt(self.args, train_raw_opt)
            self.train_dataset = create_dataset(train_opt)
            self.val_datasets = self._build_phase_datasets('val', fallback_phase='test')
            if not self.val_datasets:
                raise ValueError('Prior config must define at least one val or test dataset for validation.')
            print(f'Train dataset: {len(self.train_dataset)} samples')
            for dataset in self.val_datasets:
                print(f'Val dataset [{dataset.opt["name"]}]: {len(dataset)} samples')

        if stage in (None, 'test'):
            self.test_datasets = self._build_phase_datasets('test', fallback_phase='val')
            if not self.test_datasets:
                raise ValueError('Prior config must define at least one test or val dataset for testing.')
            for dataset in self.test_datasets:
                print(f'Test dataset [{dataset.opt["name"]}]: {len(dataset)} samples')

    def _build_loader(self, dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=self.args.num_workers > 0
        )

    def train_dataloader(self):
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        loaders = [self._build_loader(dataset, shuffle=False, drop_last=False) for dataset in self.val_datasets]
        return loaders[0] if len(loaders) == 1 else loaders

    def test_dataloader(self):
        loaders = [self._build_loader(dataset, shuffle=False, drop_last=False) for dataset in self.test_datasets]
        return loaders[0] if len(loaders) == 1 else loaders


def _get_accelerator_settings(args):
    if torch.cuda.is_available():
        return 'gpu', [args.gpu], 'bf16-mixed'
    return 'cpu', 1, 32


def _print_model_summary(model):
    print('\nModel Structure:')
    print(model.model)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'\nTotal Parameters: {params_m:.2f} Million')


def _build_trainer(args, callbacks=None):
    accelerator, devices, precision = _get_accelerator_settings(args)
    return pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks or [],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision=precision,
        logger=False,
        num_sanity_val_steps=0,
    )


def _load_model_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = normalize_prior_router_state_dict(ckpt['state_dict'])
        model.load_state_dict(state_dict, strict=True)
        return
    if 'params' in ckpt:
        state_dict = normalize_prior_router_state_dict(ckpt['params'])
        model.load_state_dict(state_dict, strict=True)
        return
    state_dict = normalize_prior_router_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=defaults.get('config'), type=str)
    parser.add_argument('--config_name', default=defaults.get('config_name'), type=str)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--gpu', default=defaults.get('gpu', 0), type=int)
    parser.add_argument('-e', '--epochs', default=defaults.get('epochs', 30), type=int)
    parser.add_argument('-b', '--batch_size', default=defaults.get('batch_size', 16), type=int)
    parser.add_argument('--patch_size', default=defaults.get('patch_size', 224), type=int)
    parser.add_argument('--crop_size', default=defaults.get('crop_size', 512), type=int)
    parser.add_argument('--lr', default=defaults.get('lr', 2e-4), type=float)
    parser.add_argument('--weight_decay', default=defaults.get('weight_decay', 1e-3), type=float)
    parser.add_argument('--adam_beta1', default=defaults.get('adam_beta1', 0.9), type=float)
    parser.add_argument('--adam_beta2', default=defaults.get('adam_beta2', 0.9), type=float)
    parser.add_argument('--eta_min', default=defaults.get('eta_min', 1e-5), type=float)
    parser.add_argument('--num_workers', default=defaults.get('num_workers', 8), type=int)
    parser.add_argument('--cache_memory', default=defaults.get('cache_memory', 0), type=int)
    parser.add_argument('--multi_label_classification',
                        default=defaults.get('multi_label_classification', False),
                        type=str2bool)
    parser.add_argument('--multi_label_threshold',
                        default=defaults.get('multi_label_threshold', 0.5),
                        type=float)
    parser.add_argument('--label_smoothing',
                        default=defaults.get('label_smoothing', 0.1),
                        type=float)
    parser.add_argument('--save_every', default=defaults.get('save_every', 10), type=int)
    parser.add_argument('--ckpt_dir', default=defaults.get('ckpt_dir', str(DEFAULT_CKPT_DIR)), type=str)
    parser.add_argument('--ckpt_path', default=defaults.get('ckpt_path'), type=str)

    parser.add_argument('--dataset_name', default=defaults.get('dataset_name', 'dataset'), type=str)
    parser.add_argument('--router_type', choices=['dpe'],
                        default=defaults.get('router_type', 'dpe'))
    parser.add_argument('--router_dim', default=defaults.get('router_dim', 384), type=int)
    parser.add_argument('--view_mode',
                        choices=VALID_VIEW_MODES,
                        default=defaults.get('view_mode'))
    parser.add_argument('--view_size',
                        default=defaults.get('view_size'),
                        type=int)
    parser.add_argument('--clip_vision_tower',
                        default=defaults.get('clip_vision_tower', 'openai/clip-vit-base-patch32'),
                        type=str)
    return parser


def parse_args_with_config(default_config=None):
    """Parse router CLI args with YAML defaults and CLI override support."""
    return parse_args_with_yaml(
        build_parser,
        default_config=default_config,
        fallback_name='router')


def _run_train(args, degrade_types):
    prepare_single_gpu_context(args.gpu)
    model = DPEModule(args=args, degrade_types=degrade_types)
    data_module = AllInOneClassificationDataModule(args, degrade_types)

    _print_model_summary(model)

    model_stem = router_checkpoint_stem(
        args.router_type,
        view_mode=args.view_mode,
        view_size=args.view_size)
    if args.multi_label_classification:
        model_stem += '_ml'
    real_ckpt_dir = config_checkpoint_dir(args.ckpt_dir, args.config_name)
    os.makedirs(real_ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(real_ckpt_dir),
        filename=f'{model_stem}_{{epoch:02d}}',
        save_top_k=-1,
        every_n_epochs=args.save_every,
        save_last=True
    )

    trainer = _build_trainer(args, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)
    if args.config:
        print(f'Config: {args.config}')
    print(f'Config Name: {args.config_name}')
    print(f'Checkpoints saved to: {real_ckpt_dir}')
    print('\nTraining completed!')


def _run_test(args, degrade_types):
    if not args.ckpt_path:
        raise ValueError('--ckpt_path is required when --mode test.')
    if not osp.exists(args.ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt_path}')

    prepare_single_gpu_context(args.gpu)
    model = DPEModule(args=args, degrade_types=degrade_types)
    _load_model_weights(model, args.ckpt_path)
    data_module = AllInOneClassificationDataModule(args, degrade_types)

    print(f'\nTesting checkpoint: {args.ckpt_path}')
    _print_model_summary(model)

    trainer = _build_trainer(args, callbacks=[])
    trainer.test(model, datamodule=data_module)


def run_training(args):
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    degrade_types = infer_prior_degrade_types(args.datasets)

    if args.cache_memory and args.num_workers > 0:
        print(
            'Warning: cache_memory=True with num_workers>0 may trigger memory leaks. '
            'The default is now disabled; only enable it deliberately.')

    if args.mode == 'test':
        _run_test(args, degrade_types)
    else:
        _run_train(args, degrade_types)


if __name__ == '__main__':
    run_training(parse_args_with_config())
