import importlib
from collections import OrderedDict
from copy import deepcopy

import torch

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel


loss_module = importlib.import_module('basicsr.models.losses')


class ImageRestorationModel(BaseModel):
    """Standard single-input image restoration model without DPE."""

    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g')
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        self.print_network(self.net_g)
        self.loss_modules, self.loss_names = loss_module.build_loss_modules(
            self.opt['train'], loss_module, self.device)
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = [
            param for param in self.net_g.parameters()
            if param.requires_grad
        ]
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                [{'params': optim_params}], **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                [{'params': optim_params}], **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'Unsupported optimizer: {optim_type}')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        del is_val
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device) if 'gt' in data else None
        self.lq_path = data.get('lq_path')
        self.gt_path = data.get('gt_path')
        self.degrade_type = data.get('degrade_type')
        self.filename = data.get('filename')

    def optimize_parameters(self, current_iter, tb_logger):
        del current_iter, tb_logger
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        l_total, loss_dict = loss_module.compute_losses(
            self.loss_modules, self.loss_names, self.output, self.gt)
        l_total.backward()
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            for i in range(0, n, m):
                pred = self.net_g(self.lq[i:i + m])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach())
            if self.opt['val'].get('grids', False):
                self.outs = outs
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
