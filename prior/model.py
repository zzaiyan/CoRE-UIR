from collections import OrderedDict

import torch
import torch.nn as nn
from anycapture import get_local

from prior.clip_encoder import CLIPTower
from prior.utils_image import normalize_view_mode, num_views_for_view_mode


class PriorRouterBase(nn.Module):
    """Base class for DPE-style degradation prior encoders."""

    router_type = None
    checkpoint_stem = None
    default_view_mode = None
    supported_view_modes = ()

    def __init__(self, dim, cls_num, *, view_mode=None, view_size=None):
        super().__init__()
        self.dim = int(dim)
        self.cls_num = int(cls_num)
        self.view_mode = normalize_view_mode(view_mode)
        self.view_size = int(view_size)

    @property
    def num_views(self):
        return num_views_for_view_mode(self.view_mode)

    def build_heads(self, input_dim, hidden_dim=None):
        hidden_dim = int(hidden_dim or self.dim * 2)
        self.de_extractor = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(self.dim, self.cls_num)

    def forward_view_features(self, x):
        raise NotImplementedError

    def aggregate_view_features(self, view_features):
        if self.view_mode in {'center', 'global'}:
            return view_features[:, 0, :]
        if self.view_mode == 'dual':
            return torch.cat([view_features[:, 0, :], view_features[:, 1, :]], dim=1)
        raise ValueError(f'Unsupported DPE view_mode: {self.view_mode}')

    @get_local('feat', 'de_prior', 'de_cls')
    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f'Expected DPE input shape (B, N, C, H, W), got {tuple(x.shape)}')
        if x.shape[1] != self.num_views:
            raise ValueError(
                f'{self.__class__.__name__} expects {self.num_views} views for '
                f'view_mode={self.view_mode}, got {x.shape[1]}')

        view_features = self.forward_view_features(x)
        feat = self.aggregate_view_features(view_features)
        de_prior = self.de_extractor(feat)
        de_cls = self.cls_head(de_prior)
        return de_cls, de_prior


class DegradationPriorEmbedding(PriorRouterBase):
    """Frozen CLIP ViT-B/32 based degradation prior embedding."""

    router_type = 'dpe'
    checkpoint_stem = 'DPE'
    default_view_mode = 'dual'
    supported_view_modes = ('center', 'global', 'dual')

    def __init__(
            self,
            vision_tower='openai/clip-vit-base-patch32',
            cls_num=6,
            dim=384,
            view_mode='dual',
            view_size=224):
        view_mode = normalize_view_mode(view_mode or self.default_view_mode)
        if view_mode not in self.supported_view_modes:
            raise ValueError(
                f'DPE does not support view_mode={view_mode}. '
                f'Supported modes: {self.supported_view_modes}')

        clip_model = CLIPTower(vision_tower)
        clip_model.requires_grad_(False)
        hidden_size = clip_model.vision_tower.visual_projection.weight.shape[0]

        super().__init__(
            dim=dim or hidden_size // 2,
            cls_num=cls_num,
            view_mode=view_mode,
            view_size=view_size)
        self.clip_model = clip_model

        input_dim = hidden_size if self.view_mode in {'center', 'global'} else hidden_size * 2
        self.build_heads(input_dim, hidden_dim=hidden_size)

    def forward_view_features(self, x):
        batch_size, num_views = x.shape[:2]
        features = self.clip_model(x.flatten(0, 1))
        return features.view(batch_size, num_views, -1)


ROUTER_REGISTRY = {
    DegradationPriorEmbedding.router_type: DegradationPriorEmbedding,
    'degradationpriorembedding': DegradationPriorEmbedding,
}


def _strip_router_state_dict_prefixes(state_dict):
    normalized = OrderedDict(state_dict)
    changed = True
    while changed:
        changed = False
        keys = list(normalized.keys())
        for prefix in ('module.', 'model.', 'student.', 'router.'):
            if any(key.startswith(prefix) for key in keys):
                normalized = OrderedDict(
                    (key[len(prefix):] if key.startswith(prefix) else key, value)
                    for key, value in normalized.items())
                changed = True
                break
    return normalized


def _remap_legacy_clip_router_keys(state_dict):
    if not any(key.startswith('clip_vision_tower.') for key in state_dict.keys()):
        return state_dict

    key_mapping = (
        ('clip_vision_tower.vision_embeddings.', 'clip_model.vision_tower.vision_model.embeddings.'),
        ('clip_vision_tower.vision_pre_layrnorm.', 'clip_model.vision_tower.vision_model.pre_layrnorm.'),
        ('clip_vision_tower.vision_encoder.', 'clip_model.vision_tower.vision_model.encoder.'),
        ('clip_vision_tower.vision_post_layernorm.', 'clip_model.vision_tower.vision_model.post_layernorm.'),
        ('clip_vision_tower.visual_projection.', 'clip_model.vision_tower.visual_projection.'),
    )
    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in key_mapping:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


def _drop_pruned_clip_text_keys(state_dict):
    text_prefixes = (
        'clip_model.text.',
        'clip_model.transformer.',
        'clip_model.token_embedding.',
        'clip_model.ln_final.',
    )
    exact_keys = {
        'clip_model.positional_embedding',
        'clip_model.text_projection',
        'clip_model.logit_scale',
        'clip_model.attn_mask',
    }
    filtered = OrderedDict()
    for key, value in state_dict.items():
        if key in exact_keys:
            continue
        if any(key.startswith(prefix) for prefix in text_prefixes):
            continue
        filtered[key] = value
    return filtered


def normalize_prior_router_state_dict(state_dict):
    state_dict = _strip_router_state_dict_prefixes(state_dict)
    state_dict = _remap_legacy_clip_router_keys(state_dict)
    state_dict = _drop_pruned_clip_text_keys(state_dict)
    return state_dict


def get_prior_router_class(router_type):
    router_key = str(router_type).lower()
    if router_key not in ROUTER_REGISTRY:
        raise ValueError(f'Unsupported prior router type: {router_type}')
    return ROUTER_REGISTRY[router_key]


def router_checkpoint_stem(router_type, **kwargs):
    del kwargs
    return get_prior_router_class(router_type).checkpoint_stem


def router_default_view_mode(router_type):
    return get_prior_router_class(router_type).default_view_mode


def router_supported_view_modes(router_type):
    return tuple(get_prior_router_class(router_type).supported_view_modes)


def build_prior_router(router_type, cls_num, **kwargs):
    router_cls = get_prior_router_class(router_type)
    return router_cls(
        vision_tower=kwargs.get('vision_tower', 'openai/clip-vit-base-patch32'),
        cls_num=cls_num,
        dim=kwargs.get('dim'),
        view_mode=kwargs.get('view_mode'),
        view_size=kwargs.get('view_size', 224))


__all__ = [
    'PriorRouterBase',
    'DegradationPriorEmbedding',
    'ROUTER_REGISTRY',
    'build_prior_router',
    'get_prior_router_class',
    'normalize_prior_router_state_dict',
    'router_default_view_mode',
    'router_checkpoint_stem',
    'router_supported_view_modes',
]
