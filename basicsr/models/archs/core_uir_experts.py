import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from anycapture import get_local

from basicsr.models.archs.arch_util import LayerNorm2d


class CoREModule(nn.Module):
    """Common-and-Residual Expert module.

    The common dense expert is always active. The low-rank residual expert
    library is routed by PG-Router logits. The implementation computes experts
    in a vectorized way for checkpoint compatibility, then applies Top-k gates.
    """

    def __init__(
            self,
            shared_expert,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            r=4,
            scaling=0.1,
            lora_dropout=0.0,
            num_experts=6,
            top_k=3,
            lora_prenorm=False):
        super().__init__()
        if not isinstance(shared_expert, nn.Module):
            raise TypeError('shared_expert must be an nn.Module.')
        if isinstance(kernel_size, tuple):
            if kernel_size[0] != kernel_size[1]:
                raise ValueError('CoREModule supports square kernels only.')
            kernel_size = kernel_size[0]
        if groups != 1:
            raise ValueError('CoREModule supports groups=1 only.')

        self.shared_expert = shared_expert
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.r = int(r)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.expert_norm = LayerNorm2d(self.in_channels) if lora_prenorm else nn.Identity()

        self.scaling_mode = scaling
        if scaling in ('auto', 'softplus', 'sigmoid'):
            self.scaling = nn.Parameter(
                torch.full((1, self.in_channels, 1, 1), 0.1),
                requires_grad=True)
        elif isinstance(scaling, (int, float)):
            self.scaling = scaling
        else:
            raise ValueError('scaling must be a number or "auto", "softplus", "sigmoid".')

        self.lora_A = nn.Conv2d(
            self.in_channels,
            self.r * self.num_experts,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=False)
        self.lora_B = nn.Conv2d(
            self.r * self.num_experts,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            groups=1,
            bias=False)
        self.reset_experts()

    def reset_experts(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def _normalized_topk_logits(self, logits):
        if logits is None:
            raise ValueError('PG-Router logits are required when CoRE experts are enabled.')
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[-1] != self.num_experts:
            raise ValueError(
                f'logits dim mismatch: got {logits.shape[-1]} experts, '
                f'expected {self.num_experts}.')
        topk_idx = torch.topk(logits, k=self.top_k, dim=-1).indices
        mask = F.one_hot(topk_idx, num_classes=self.num_experts).any(dim=-2)
        routed_logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        return routed_logits.softmax(dim=-1)

    def _parallel_expert_conv(self, x, gates):
        batch_size = x.shape[0]
        a_out = self.lora_A(self.dropout(x))
        _, _, out_h, out_w = a_out.shape
        a_out = a_out.view(batch_size, self.num_experts, self.r, out_h, out_w)
        a_out = a_out * gates.view(batch_size, self.num_experts, 1, 1, 1)
        a_out = a_out.view(batch_size, self.num_experts * self.r, out_h, out_w)
        return self.lora_B(a_out)

    def _scaling_alpha(self):
        if self.scaling_mode == 'softplus':
            return F.softplus(self.scaling)
        if self.scaling_mode == 'sigmoid':
            return torch.sigmoid(self.scaling)
        return self.scaling

    @get_local(
        'core_in',
        'shared_out',
        'dynamic_out',
        'core_out',
        'route_logits',
        'route_weights',
        'route_mask')
    def forward(self, x, route_logits=None):
        core_in = x
        shared_out = self.shared_expert(x)
        route_weights = self._normalized_topk_logits(route_logits)
        route_mask = route_weights > 0
        dynamic_out = self._parallel_expert_conv(
            self.expert_norm(x), route_weights) * self._scaling_alpha()
        core_out = shared_out + dynamic_out
        return core_out
