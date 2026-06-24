# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel

from prior.model import build_prior_router, normalize_prior_router_state_dict


loss_module = importlib.import_module('basicsr.models.losses')
class IRPriorModel(BaseModel):
    """Image restoration model conditioned on a frozen DPE prior."""

    def __init__(self, opt):
        super(IRPriorModel, self).__init__(opt)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        opt_prior = deepcopy(opt['network_prior'])
        prior_type = opt_prior.pop('type', 'DegradationPriorEmbedding')
        self.net_prior = build_prior_router(
            router_type=prior_type,
            cls_num=opt_prior.pop('cls_num'),
            dim=opt_prior.pop('dim', None),
            view_mode=opt_prior.pop('view_mode', None),
            view_size=opt_prior.pop('view_size', None),
            vision_tower=opt_prior.pop('vision_tower', 'openai/clip-vit-base-patch32'))
        self.net_prior.to(dtype=torch.bfloat16)

        ckpt_path = opt_prior.pop('ckpt', None)
        if ckpt_path is not None:
            ckpt = self._torch_load_compat(ckpt_path, map_location='cpu', weights_only=True)
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            elif 'params' in ckpt:
                ckpt = ckpt['params']
            ckpt = normalize_prior_router_state_dict(ckpt)
            self.net_prior.load_state_dict(ckpt, strict=True)
            print(f"Load DPE state_dict from {ckpt_path}")

        self.net_prior = self.model_to_device(self.net_prior)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        self.print_network(self.net_g)
        self.net_prior.eval()
        for param in self.net_prior.parameters():
            param.requires_grad = False
        self.loss_modules, self.loss_names = loss_module.build_loss_modules(
            self.opt['train'], loss_module, self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        del is_val
        self.lq = data['lq'].to(self.device)
        if 'lq_clip' in data:
            self.lq_clip = data['lq_clip'].to(self.device)
        else:
            self.lq_clip = None
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None
        self.lq_path = data.get('lq_path')
        self.gt_path = data.get('gt_path')
        self.degrade_type = data.get('degrade_type')
        self.filename = data.get('filename')

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, de_prior = self.net_prior(self.lq_clip)
            de_prior = de_prior.float()
        preds = self.net_g(self.lq, de_prior)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total, loss_dict = loss_module.compute_losses(
            self.loss_modules, self.loss_names, self.output, self.gt)

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        """Run inference for the current batch.

        When grid validation is enabled, predictions are stored in `self.outs`
        for later merging by `BaseModel.grids_inverse()`. Otherwise the full
        batch prediction is concatenated into `self.output`.
        """
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    de_prior = self.net_prior(self.lq_clip[i:j])[1]
                    de_prior = de_prior.float()
                pred = self.net_g(self.lq[i:j], de_prior)
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach())
                i = j

            # Check if using grids mode
            if self.opt['val'].get('grids', False):
                self.outs = outs    # outs = (b, c, h, w), b==1
            else:
                self.output = torch.cat(outs, dim=0)
        self.net_g.train()

        return 0.

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
