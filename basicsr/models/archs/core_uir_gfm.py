import torch
import torch.nn as nn
from anycapture import get_local

from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.core_uir_router import PriorModulatorBase


class StateGFM(PriorModulatorBase):
    """State-aware GFM that conditions on both features and DPE prior."""

    def __init__(self, chan, prior_dim, reduction=16, bias=True, fusion='add'):
        super().__init__()
        self.fusion = str(fusion).lower()
        if self.fusion not in {'add', 'concat'}:
            raise ValueError(f'Unsupported StateGFM fusion: {fusion}. Expected "add" or "concat".')
        hidden_dim = max(1, chan // reduction)
        fused_dim = chan if self.fusion == 'add' else chan * 2

        self.norm = LayerNorm2d(chan)
        self.alpha = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)
        self.to_state = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(chan, chan, bias=bias),
        )
        self.to_prior = nn.Sequential(nn.Linear(prior_dim, chan, bias=bias))
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, chan, bias=bias),
            nn.Sigmoid(),
        )

    @get_local('gfm_in', 'state', 'prior', 'fused', 'gate', 'gfm_out')
    def forward(self, x, prior_feat):
        gfm_in = x
        state = self.to_state(x)
        prior = self.to_prior(prior_feat)
        fused = torch.cat([state, prior], dim=1) if self.fusion == 'concat' else state + prior
        gate = self.fc(fused).unsqueeze(-1).unsqueeze(-1)
        gfm_out = x + self.alpha * (self.norm(x) * gate)
        return gfm_out


class StateOnlyGFM(PriorModulatorBase):
    def __init__(self, chan, prior_dim, reduction=16, bias=True):
        del prior_dim
        super().__init__()
        hidden_dim = max(1, chan // reduction)
        self.norm = LayerNorm2d(chan)
        self.alpha = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)
        self.to_state = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(chan, chan, bias=bias),
        )
        self.fc = nn.Sequential(
            nn.Linear(chan, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, chan, bias=bias),
            nn.Sigmoid(),
        )

    @get_local('gfm_in', 'state', 'gate', 'gfm_out')
    def forward(self, x, prior_feat):
        del prior_feat
        gfm_in = x
        state = self.to_state(x)
        gate = self.fc(state).unsqueeze(-1).unsqueeze(-1)
        gfm_out = x + self.alpha * (self.norm(x) * gate)
        return gfm_out


class PriorOnlyGFM(PriorModulatorBase):
    def __init__(self, chan, prior_dim, reduction=16, bias=True):
        super().__init__()
        hidden_dim = max(1, chan // reduction)
        self.norm = LayerNorm2d(chan)
        self.alpha = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)
        self.to_prior = nn.Sequential(nn.Linear(prior_dim, chan, bias=bias))
        self.fc = nn.Sequential(
            nn.Linear(chan, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, chan, bias=bias),
            nn.Sigmoid(),
        )

    @get_local('gfm_in', 'prior', 'gate', 'gfm_out')
    def forward(self, x, prior_feat):
        gfm_in = x
        prior = self.to_prior(prior_feat)
        gate = self.fc(prior).unsqueeze(-1).unsqueeze(-1)
        gfm_out = x + self.alpha * (self.norm(x) * gate)
        return gfm_out
