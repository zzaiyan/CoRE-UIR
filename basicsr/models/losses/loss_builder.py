from collections import OrderedDict
from copy import deepcopy

from torch import nn


def build_loss_modules(train_opt, loss_module, device):
    loss_opts = deepcopy(train_opt.get('losses'))
    if not loss_opts:
        raise ValueError('train.losses must contain at least one loss entry.')
    if not isinstance(loss_opts, list):
        raise TypeError('train.losses must be a list of loss configs.')

    loss_modules = nn.ModuleDict()
    loss_names = []
    for index, loss_opt in enumerate(loss_opts):
        if not isinstance(loss_opt, dict):
            raise TypeError('Each item in train.losses must be a dict.')

        current_opt = deepcopy(loss_opt)
        loss_type = current_opt.pop('type', None)
        if loss_type is None:
            raise KeyError(f'train.losses[{index}] is missing required key: type')

        loss_name = current_opt.pop('name', f'loss_{index + 1}')
        if loss_name in loss_modules:
            raise ValueError(f'Duplicate loss name found in train.losses: {loss_name}')

        loss_cls = getattr(loss_module, loss_type, None)
        if loss_cls is None:
            raise ValueError(f'Loss {loss_type} is not found.')

        loss_modules[loss_name] = loss_cls(**current_opt).to(device)
        loss_names.append(loss_name)

    return loss_modules, loss_names


def compute_losses(loss_modules, loss_names, pred, target):
    total_loss = pred.new_zeros(())
    loss_dict = OrderedDict()

    for loss_name in loss_names:
        result = loss_modules[loss_name](pred, target)
        if isinstance(result, tuple):
            main_loss = result[0]
            aux_losses = result[1:]
        else:
            main_loss = result
            aux_losses = ()

        if main_loss is not None:
            total_loss = total_loss + main_loss
            loss_dict[loss_name] = main_loss

        for index, aux_loss in enumerate(aux_losses, start=1):
            if aux_loss is None:
                continue
            aux_name = f'{loss_name}_aux{index}'
            total_loss = total_loss + aux_loss
            loss_dict[aux_name] = aux_loss

    loss_dict['l_total'] = total_loss
    return total_loss, loss_dict