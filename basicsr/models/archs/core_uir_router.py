import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from anycapture import get_local

from basicsr.models.archs.arch_util import LayerNorm2d


class PriorAdapterBase(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim


class IdentityPriorAdapter(PriorAdapterBase):
    def __init__(self, prior_dim=384):
        super().__init__(out_dim=prior_dim)

    def forward(self, prior):
        return prior


class LinearPriorAdapter(PriorAdapterBase):
    def __init__(self, prior_dim=384, out_dim=128, use_norm=True, bias=True):
        super().__init__(out_dim=out_dim)
        self.norm = nn.LayerNorm(prior_dim) if use_norm else nn.Identity()
        self.proj = nn.Linear(prior_dim, out_dim, bias=bias)

    def forward(self, prior):
        return self.proj(self.norm(prior))


class MLPPriorAdapter(PriorAdapterBase):
    def __init__(self, prior_dim=384, hidden_dim=128, out_dim=None, use_norm=True, bias=True):
        out_dim = hidden_dim if out_dim is None else out_dim
        super().__init__(out_dim=out_dim)
        self.norm = nn.LayerNorm(prior_dim) if use_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
            nn.GELU(),
        )

    def forward(self, prior):
        return self.net(self.norm(prior))


class RouterHeadBase(nn.Module):
    def __init__(self, prior_dim, num_experts):
        super().__init__()
        self.prior_dim = prior_dim
        self.num_experts = num_experts


class LinearRouterHead(RouterHeadBase):
    def __init__(self, prior_dim, num_experts, bias=True):
        super().__init__(prior_dim, num_experts)
        self.proj = nn.Linear(prior_dim, num_experts, bias=bias)

    def forward(self, prior_feat):
        return self.proj(prior_feat)


class MLPRouterHead(RouterHeadBase):
    def __init__(self, prior_dim, num_experts, hidden_dim=None, bias=True):
        hidden_dim = prior_dim if hidden_dim is None else hidden_dim
        super().__init__(prior_dim, num_experts)
        self.net = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts, bias=bias),
        )

    def forward(self, prior_feat):
        return self.net(prior_feat)


class PrototypeRouterHead(RouterHeadBase):
    """Prototype-guided router based on cosine similarity."""

    def __init__(self, prior_dim, num_experts, init_temperature=10.0):
        super().__init__(prior_dim, num_experts)
        self.prototypes = nn.Parameter(torch.randn(num_experts, prior_dim))
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(max(init_temperature, 1e-3)), dtype=torch.float32))

    def forward(self, prior_feat):
        prior_feat = F.normalize(prior_feat, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.matmul(prior_feat, prototypes.t()) * scale


class PriorModulatorBase(nn.Module):
    def forward(self, x, prior_feat):
        raise NotImplementedError


class NoGFM(PriorModulatorBase):
    def forward(self, x, prior_feat):
        del prior_feat
        return x


class GateGFM(PriorModulatorBase):
    def __init__(self, chan, prior_dim, bias=True):
        super().__init__()
        self.norm = LayerNorm2d(chan)
        self.alpha = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)
        self.to_gate = nn.Linear(prior_dim, chan, bias=bias)

    @get_local('gfm_in', 'gate', 'gfm_out')
    def forward(self, x, prior_feat):
        gfm_in = x
        gate = torch.sigmoid(self.to_gate(prior_feat)).unsqueeze(-1).unsqueeze(-1)
        gfm_out = x + self.alpha * (self.norm(x) * gate)
        return gfm_out


class FiLMGFM(PriorModulatorBase):
    def __init__(self, chan, prior_dim, bias=True):
        super().__init__()
        self.norm = LayerNorm2d(chan)
        self.alpha = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)
        self.to_affine = nn.Linear(prior_dim, chan * 2, bias=bias)

    @get_local('gfm_in', 'gamma', 'beta', 'gfm_out')
    def forward(self, x, prior_feat):
        gfm_in = x
        gamma, beta = self.to_affine(prior_feat).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        mod = self.norm(x) * gamma + beta
        gfm_out = x + self.alpha * mod
        return gfm_out
