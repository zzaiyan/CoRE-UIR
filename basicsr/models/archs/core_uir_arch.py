"""Paper-facing CoRE-UIR architecture entry points.

The implementation intentionally keeps the internal module names of the
validated development model so existing checkpoints remain loadable. Public
configuration files should instantiate :class:`CoREUIR`.
"""

from basicsr.models.archs.core_uir_router import PrototypeRouterHead
from basicsr.models.archs.core_uir_gfm import StateGFM
from basicsr.models.archs.core_uir_backbone import LocalGLANetBackbone, UnifiedRouteBlock
from basicsr.models.archs.core_uir_experts import CoREModule


class CoREUIR(LocalGLANetBackbone):
    """CoRE-UIR / GLA-Net restoration backbone.

    Defaults follow the paper setting:
    CLIP DPE embedding dimension 384, stage-level PG-Router, State GFM,
    six low-rank residual experts and Top-3 routing.
    """

    def __init__(
            self,
            img_channel=3,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=(1, 1, 1, 28),
            dec_blk_nums=(1, 1, 1, 1),
            train_size=(1, 3, 256, 256),
            r=4,
            num_experts=6,
            top_k=3,
            lora_scaling='auto',
            lora_prenorm=False,
            final_skip=True,
            prior_dim=384,
            prior_adapter_type='MLP',
            prior_hidden_dim=384,
            router_head_type='prototype',
            router_hidden_dim=None,
            router_temperature=10.0,
            gfm_type='state',
            gfm_mid_chans=128,
            gfm_reduction=16,
            gfm_bias=True,
            gfm_fusion='add',
            route_level='stage',
            expert_structure='asymmetric_moe',
            use_gfm=True,
            use_mole=True,
            gfm_positions=('encoder', 'mid', 'decoder'),
            mole_positions=('encoder', 'mid', 'decoder'),
            fast_imp=False):
        super().__init__(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=list(enc_blk_nums),
            dec_blk_nums=list(dec_blk_nums),
            train_size=tuple(train_size),
            r=r,
            num_experts=num_experts,
            top_k=top_k,
            lora_scaling=lora_scaling,
            lora_prenorm=lora_prenorm,
            final_skip=final_skip,
            prior_dim=prior_dim,
            prior_adapter_type=prior_adapter_type,
            prior_hidden_dim=prior_hidden_dim,
            router_head_type=router_head_type,
            router_hidden_dim=router_hidden_dim,
            router_temperature=router_temperature,
            gfm_type=gfm_type,
            gfm_mid_chans=gfm_mid_chans,
            gfm_reduction=gfm_reduction,
            gfm_bias=gfm_bias,
            gfm_fusion=gfm_fusion,
            route_level=route_level,
            expert_structure=expert_structure,
            use_gfm=use_gfm,
            use_mole=use_mole,
            gfm_positions=gfm_positions,
            mole_positions=mole_positions,
            fast_imp=fast_imp)


# Paper-facing aliases used by docs, smoke tests, and downstream scripts.
PGRouter = PrototypeRouterHead
GFM = StateGFM
CoREBlock = UnifiedRouteBlock
LowRankResidualExpert = CoREModule


__all__ = [
    'CoREUIR',
    'PGRouter',
    'GFM',
    'CoREBlock',
    'LowRankResidualExpert',
]
