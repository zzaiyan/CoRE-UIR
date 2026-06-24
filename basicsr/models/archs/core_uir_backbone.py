from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base3
from basicsr.models.archs.core_uir_experts import CoREModule
from basicsr.models.archs.core_uir_router import (
    IdentityPriorAdapter,
    LinearPriorAdapter,
    LinearRouterHead,
    MLPRouterHead,
    MLPPriorAdapter,
    NoGFM,
    PrototypeRouterHead,
    GateGFM,
    FiLMGFM,
)
from basicsr.models.archs.core_uir_blocks import PlainNAFBlock, SimpleGate
from basicsr.models.archs.core_uir_gfm import (
    PriorOnlyGFM,
    StateGFM,
    StateOnlyGFM,
)


def _normalize_route_level(route_level):
    route_level = str(route_level).lower()
    if route_level == 'sample':
        return 'sample'
    if route_level == 'stage':
        return 'stage'
    if route_level == 'block':
        return 'block'
    raise ValueError(f'Unsupported route_level: {route_level}')


def _normalize_positions(positions, default=('encoder', 'mid', 'decoder')):
    if positions is None:
        positions = default
    if isinstance(positions, str):
        text = positions.strip().lower()
        if text in {'', 'none'}:
            return tuple()
        if text == 'all':
            return ('encoder', 'mid', 'decoder')
        positions = [item.strip() for item in text.split(',') if item.strip()]
    normalized = []
    for item in positions:
        item = str(item).strip().lower()
        if item == 'encoder':
            item = 'encoder'
        elif item == 'mid':
            item = 'mid'
        elif item == 'decoder':
            item = 'decoder'
        else:
            raise ValueError(f'Unsupported position token: {item}')
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _normalize_expert_structure(expert_structure):
    expert_structure = str(expert_structure).strip().lower()
    if expert_structure in {'shared_backbone', 'asymmetric_moe', 'standard_moe_ffn'}:
        return expert_structure
    raise ValueError(f'Unsupported expert_structure: {expert_structure}')


class PriorRouterFactoryMixin:
    @staticmethod
    def build_prior_adapter(prior_adapter_type, prior_dim=384, prior_hidden_dim=128):
        prior_adapter_type = str(prior_adapter_type).lower()
        if prior_adapter_type == 'identity':
            return IdentityPriorAdapter(prior_dim=prior_dim)
        if prior_adapter_type == 'linear':
            return LinearPriorAdapter(
                prior_dim=prior_dim,
                out_dim=prior_hidden_dim,
                use_norm=True)
        if prior_adapter_type == 'mlp':
            return MLPPriorAdapter(
                prior_dim=prior_dim,
                hidden_dim=prior_hidden_dim,
                out_dim=prior_hidden_dim,
                use_norm=True)
        raise ValueError(f'Unsupported prior_adapter_type: {prior_adapter_type}')

    @staticmethod
    def build_router_head(router_head_type, prior_dim, num_experts,
                          router_hidden_dim=None, router_temperature=10.0):
        router_head_type = str(router_head_type).lower()
        if router_head_type == 'linear':
            return LinearRouterHead(prior_dim=prior_dim, num_experts=num_experts)
        if router_head_type == 'prototype':
            return PrototypeRouterHead(
                prior_dim=prior_dim,
                num_experts=num_experts,
                init_temperature=router_temperature)
        if router_head_type == 'mlp':
            return MLPRouterHead(
                prior_dim=prior_dim,
                num_experts=num_experts,
                hidden_dim=router_hidden_dim)
        raise ValueError(f'Unsupported router_head_type: {router_head_type}')

    @staticmethod
    def build_prior_modulator(gfm_type, chan, prior_dim, mid_chans=128,
                              reduction=16, bias=True, fusion='add'):
        gfm_type = str(gfm_type).lower()
        if gfm_type == 'none':
            return NoGFM()
        if gfm_type == 'gate':
            return GateGFM(chan=chan, prior_dim=prior_dim, bias=bias)
        if gfm_type == 'film':
            return FiLMGFM(chan=chan, prior_dim=prior_dim, bias=bias)
        if gfm_type == 'state':
            return StateGFM(
                chan=chan,
                prior_dim=prior_dim,
                reduction=reduction,
                bias=bias,
                fusion=fusion)
        if gfm_type == 'state_only':
            return StateOnlyGFM(
                chan=chan,
                prior_dim=prior_dim,
                reduction=reduction,
                bias=bias)
        if gfm_type == 'prior_only':
            return PriorOnlyGFM(
                chan=chan,
                prior_dim=prior_dim,
                reduction=reduction,
                bias=bias)
        raise ValueError(f'Unsupported gfm_type: {gfm_type}')


class StandardFFNExpert(nn.Module):
    """A standalone copy of the NAFBlock FFN branch used by standard MoE."""

    def __init__(self, c, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        ffn_channel = FFN_Expand * c
        self.conv1 = nn.Conv2d(c, ffn_channel, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        return self.dropout(x)


class StandardMoEFFNBlock(nn.Module):
    """Shared spatial trunk + routed full-rank FFN experts."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                 num_experts=3, top_k=1):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3,
                               stride=1, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1,
                               stride=1, padding=0, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1,
                      stride=1, padding=0, bias=True),
        )
        self.sg = SimpleGate()

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.experts = nn.ModuleList([
            StandardFFNExpert(c, FFN_Expand=FFN_Expand, drop_out_rate=drop_out_rate)
            for _ in range(self.num_experts)
        ])

    def _normalize_topk_logits(self, logits):
        if logits is None:
            raise ValueError('StandardMoEFFNBlock requires route_logits when use_mole=True.')
        if logits.shape[-1] != self.num_experts:
            raise ValueError(
                f'logits dim mismatch: got {logits.shape[-1]} experts, '
                f'expected {self.num_experts}.')
        topk_idx = torch.topk(logits, k=self.top_k, dim=-1).indices
        mask = F.one_hot(topk_idx, num_classes=self.num_experts).any(dim=-2)
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        return torch.softmax(masked_logits, dim=-1)

    def forward(self, inp, route_logits):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        ffn_in = self.norm2(y)
        route_weights = self._normalize_topk_logits(route_logits)
        expert_outs = torch.stack([expert(ffn_in) for expert in self.experts], dim=1)
        mixed = torch.sum(
            expert_outs * route_weights[:, :, None, None, None],
            dim=1)
        return y + mixed * self.gamma

class UnifiedRouteBlock(nn.Module):
    LORA_SCALING = 0.1

    def __init__(self, c, use_mole=True, route_level='sample', router_template=None,
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., r=4,
                 num_experts=3, top_k=1, lora_prenorm=True,
                 expert_structure='asymmetric_moe'):
        super().__init__()
        self.use_mole = bool(use_mole)
        self.route_level = _normalize_route_level(route_level)
        self.expert_structure = _normalize_expert_structure(expert_structure)

        if self.use_mole and self.expert_structure == 'asymmetric_moe':
            self.mole_block = CoREModule(
                shared_expert=PlainNAFBlock(c, DW_Expand, FFN_Expand, drop_out_rate),
                in_channels=c,
                out_channels=c,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                r=r,
                scaling=self.LORA_SCALING,
                num_experts=num_experts,
                top_k=top_k,
                lora_prenorm=lora_prenorm,
            )
            self.standard_moe_ffn = None
        elif self.use_mole and self.expert_structure == 'standard_moe_ffn':
            self.standard_moe_ffn = StandardMoEFFNBlock(
                c,
                DW_Expand=DW_Expand,
                FFN_Expand=FFN_Expand,
                drop_out_rate=drop_out_rate,
                num_experts=num_experts,
                top_k=top_k,
            )
            self.mole_block = None
        else:
            self.shared_block = PlainNAFBlock(c, DW_Expand, FFN_Expand, drop_out_rate)
            self.mole_block = None
            self.standard_moe_ffn = None

        if self.use_mole and self.route_level == 'block':
            if router_template is None:
                raise ValueError('Block-level routing requires router_template.')
            self.router_head = deepcopy(router_template)
        else:
            self.router_head = None

    def forward(self, inp, prior_feat=None, route_logits=None):
        if not self.use_mole:
            return self.shared_block(inp)

        if self.router_head is not None:
            route_logits = self.router_head(prior_feat)
        if self.mole_block is not None:
            return self.mole_block(inp, route_logits)
        if self.standard_moe_ffn is not None:
            return self.standard_moe_ffn(inp, route_logits)
        raise RuntimeError(f'Unsupported expert_structure: {self.expert_structure}')


class UnifiedPriorStage(nn.Module):
    def __init__(self, chan, num_blocks, prior_modulator, use_mole=True,
                 route_level='sample', router_template=None, r=4, num_experts=3,
                 top_k=1, lora_prenorm=True, expert_structure='asymmetric_moe'):
        super().__init__()
        self.prior_modulator = prior_modulator
        self.blocks = nn.ModuleList([
            UnifiedRouteBlock(
                chan,
                use_mole=use_mole,
                route_level=route_level,
                router_template=router_template,
                r=r,
                num_experts=num_experts,
                top_k=top_k,
                lora_prenorm=lora_prenorm,
                expert_structure=expert_structure)
            for _ in range(num_blocks)
        ])

        route_level = _normalize_route_level(route_level)
        if use_mole and route_level == 'stage':
            if router_template is None:
                raise ValueError('Stage-level routing requires router_template.')
            self.router_head = deepcopy(router_template)
        else:
            self.router_head = None

    def forward(self, x, prior_feat, route_logits=None):
        if isinstance(self.prior_modulator, nn.Identity):
            x = self.prior_modulator(x)
        else:
            x = self.prior_modulator(x, prior_feat)

        if self.router_head is not None:
            route_logits = self.router_head(prior_feat)

        for block in self.blocks:
            x = block(x, prior_feat, route_logits)
        return x


class GLANetBackbone(nn.Module, PriorRouterFactoryMixin):
    """GLA-Net backbone with State GFM, PG-Router, and CoRE blocks."""

    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        r=4,
        num_experts=3,
        top_k=1,
        lora_scaling=0.1,
        lora_prenorm=False,
        final_skip=True,
        prior_dim=384,
        prior_adapter_type='linear',
        prior_hidden_dim=128,
        router_head_type='prototype',
        router_hidden_dim=None,
        router_temperature=10.0,
        gfm_type='film',
        gfm_mid_chans=128,
        gfm_reduction=16,
        gfm_bias=True,
        gfm_fusion='add',
        route_level='sample',
        expert_structure='asymmetric_moe',
        use_gfm=True,
        use_mole=True,
        gfm_positions=('encoder', 'mid', 'decoder'),
        mole_positions=('encoder', 'mid', 'decoder'),
    ):
        super().__init__()
        UnifiedRouteBlock.LORA_SCALING = lora_scaling
        self.final_skip = bool(final_skip)
        self.prior_dim = prior_dim
        self.route_level = _normalize_route_level(route_level)
        self.expert_structure = _normalize_expert_structure(expert_structure)
        self.use_gfm = bool(use_gfm)
        self.use_mole = bool(use_mole)
        self.gfm_fusion = str(gfm_fusion).lower()

        self.gfm_positions = _normalize_positions(
            gfm_positions,
            default=('encoder', 'mid', 'decoder') if self.use_gfm else tuple())
        self.mole_positions = _normalize_positions(
            mole_positions,
            default=('encoder', 'mid', 'decoder') if self.use_mole else tuple())

        self.intro = nn.Conv2d(
            img_channel, width, kernel_size=3, stride=1, padding=1, bias=True)
        self.ending = nn.Conv2d(
            width, img_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.prior_adapter = self.build_prior_adapter(
            prior_adapter_type,
            prior_dim=prior_dim,
            prior_hidden_dim=prior_hidden_dim)

        route_template = None
        if self.use_mole and self.mole_positions:
            route_template = self.build_router_head(
                router_head_type,
                prior_dim=self.prior_adapter.out_dim,
                num_experts=num_experts,
                router_hidden_dim=router_hidden_dim,
                router_temperature=router_temperature)

        if self.route_level == 'sample':
            self.router_head = route_template
        else:
            self.router_head = None

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(self._build_stage(
                stage_name='encoder',
                chan=chan,
                num_blocks=num,
                router_template=route_template,
                gfm_type=gfm_type,
                gfm_mid_chans=gfm_mid_chans,
                gfm_reduction=gfm_reduction,
                gfm_bias=gfm_bias,
                r=r,
                num_experts=num_experts,
                top_k=top_k,
                lora_prenorm=lora_prenorm,
                expert_structure=self.expert_structure,
            ))
            self.downs.append(
                nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2, bias=True))
            chan = chan * 2

        self.middle_blks = self._build_stage(
            stage_name='mid',
            chan=chan,
            num_blocks=middle_blk_num,
            router_template=route_template,
            gfm_type=gfm_type,
            gfm_mid_chans=gfm_mid_chans,
            gfm_reduction=gfm_reduction,
            gfm_bias=gfm_bias,
            r=r,
            num_experts=num_experts,
            top_k=top_k,
            lora_prenorm=lora_prenorm,
            expert_structure=self.expert_structure,
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(self._build_stage(
                stage_name='decoder',
                chan=chan,
                num_blocks=num,
                router_template=route_template,
                gfm_type=gfm_type,
                gfm_mid_chans=gfm_mid_chans,
                gfm_reduction=gfm_reduction,
                gfm_bias=gfm_bias,
                r=r,
                num_experts=num_experts,
                top_k=top_k,
                lora_prenorm=lora_prenorm,
                expert_structure=self.expert_structure,
            ))

        self.padder_size = 2 ** len(self.encoders)

    def _build_stage(self, stage_name, chan, num_blocks, router_template,
                     gfm_type, gfm_mid_chans,
                     gfm_reduction, gfm_bias, r, num_experts, top_k,
                     lora_prenorm, expert_structure):
        use_stage_gfm = self.use_gfm and stage_name in self.gfm_positions
        use_stage_mole = self.use_mole and stage_name in self.mole_positions
        prior_modulator = (
            self.build_prior_modulator(
                gfm_type,
                chan=chan,
                prior_dim=self.prior_adapter.out_dim,
                mid_chans=gfm_mid_chans,
                reduction=gfm_reduction,
                bias=gfm_bias,
                fusion=self.gfm_fusion)
            if use_stage_gfm else nn.Identity()
        )
        return UnifiedPriorStage(
            chan,
            num_blocks,
            prior_modulator=prior_modulator,
            use_mole=use_stage_mole,
            route_level=self.route_level,
            router_template=router_template,
            r=r,
            num_experts=num_experts,
            top_k=top_k,
            lora_prenorm=lora_prenorm,
            expert_structure=expert_structure,
        )

    def prepare_prior(self, prior):
        prior_feat = self.prior_adapter(prior)
        if self.router_head is None:
            return prior_feat, None
        route_logits = self.router_head(prior_feat)
        return prior_feat, route_logits

    def forward(self, inp, prior, probs=None):
        del probs  # kept for drop-in compatibility with 3-input wrappers

        _, _, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        prior = prior.float()
        prior_feat, route_logits = self.prepare_prior(prior)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, prior_feat, route_logits)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x, prior_feat, route_logits)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, prior_feat, route_logits)

        x = self.ending(x)
        if self.final_skip:
            x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


class LocalGLANetBackbone(Local_Base3, GLANetBackbone):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base3.__init__(self)
        GLANetBackbone.__init__(self, *args, **kwargs)

        _, _, H, W = train_size
        base_size = (int(H * 2), int(W * 2))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
