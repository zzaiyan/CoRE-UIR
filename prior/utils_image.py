from __future__ import annotations

import random
from typing import Iterable

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VALID_VIEW_MODES = ('center', 'global', 'dual')
_VIEW_COUNT = {
    'center': 1,
    'global': 1,
    'dual': 2,
}


def normalize_view_mode(view_mode: str | None) -> str | None:
    if view_mode is None:
        return None
    mode = str(view_mode).strip().lower()
    if mode not in VALID_VIEW_MODES:
        raise ValueError(
            f'Unsupported view_mode: {view_mode}. '
            f'Expected one of {VALID_VIEW_MODES}.')
    return mode


def num_views_for_view_mode(view_mode: str) -> int:
    mode = normalize_view_mode(view_mode)
    assert mode is not None
    return _VIEW_COUNT[mode]


def _default_view_size_for_mode(view_mode: str) -> int:
    mode = normalize_view_mode(view_mode)
    assert mode is not None
    return 224


def resolve_router_view_config(
        *,
        view_mode: str | None = None,
        view_size: int | None = None,
        default_view_mode: str | None = None,
        default_view_size: int | None = None) -> tuple[str, int]:
    """Resolve DPE view settings into a single `(view_mode, view_size)` pair."""
    mode = normalize_view_mode(view_mode)
    if mode is None:
        mode = normalize_view_mode(default_view_mode) or 'dual'

    size = view_size
    if size is None:
        size = int(default_view_size or _default_view_size_for_mode(mode))

    size = int(size)
    if size <= 0:
        raise ValueError(f'view_size must be a positive integer, but got {size}.')

    return mode, size


class PriorViewTransformMixin:
    """Shared CLIP-style view preprocessing for all prior routers."""

    interpolation = InterpolationMode.BICUBIC
    clip_mean = CLIP_MEAN
    clip_std = CLIP_STD

    def validate_view_config(
            self,
            view_mode: str,
            view_size: int,
            supported_view_modes: Iterable[str] | None = None) -> tuple[str, int]:
        mode = normalize_view_mode(view_mode)
        assert mode is not None
        size = int(view_size)
        if size <= 0:
            raise ValueError(f'view_size must be positive, but got {view_size}.')

        if supported_view_modes is not None:
            normalized_supported = {normalize_view_mode(item) for item in supported_view_modes}
            if mode not in normalized_supported:
                raise ValueError(
                    f'{self.__class__.__name__} does not support view_mode={mode}. '
                    f'Supported modes: {sorted(normalized_supported)}')

        return mode, size

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return transforms.functional.normalize(
            tensor,
            mean=self.clip_mean,
            std=self.clip_std)

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = transforms.functional.to_tensor(image)
        return self._normalize_tensor(tensor)

    def _resize_square(self, image: Image.Image, size: int) -> Image.Image:
        return transforms.functional.resize(
            image,
            [size, size],
            interpolation=self.interpolation)

    def _center_view_train(self, image: Image.Image, size: int) -> torch.Tensor:
        crop = transforms.RandomCrop(size)(image)
        if random.random() > 0.5:
            crop = transforms.functional.hflip(crop)
        if random.random() > 0.5:
            crop = transforms.functional.vflip(crop)
        angle = random.uniform(-90.0, 90.0)
        crop = transforms.functional.rotate(
            crop,
            angle,
            interpolation=self.interpolation)
        return self._to_tensor(crop)

    def _center_view_inference(self, image: Image.Image, size: int) -> torch.Tensor:
        crop = transforms.functional.center_crop(image, [size, size])
        return self._to_tensor(crop)

    def _global_view(self, image: Image.Image, size: int) -> torch.Tensor:
        return self._to_tensor(self._resize_square(image, size))

    def build_router_views(
            self,
            image: Image.Image,
            view_mode: str,
            view_size: int,
            *,
            phase: str = 'inference') -> torch.Tensor:
        mode, size = self.validate_view_config(view_mode, view_size)
        phase = str(phase).strip().lower()
        is_train = phase == 'train'

        if mode == 'center':
            view = self._center_view_train(image, size) if is_train else self._center_view_inference(image, size)
            return view.unsqueeze(0)
        if mode == 'global':
            return self._global_view(image, size).unsqueeze(0)
        if mode == 'dual':
            center = self._center_view_train(image, size) if is_train else self._center_view_inference(image, size)
            global_view = self._global_view(image, size)
            return torch.stack([center, global_view], dim=0)
        raise ValueError(f'Unsupported view_mode: {mode}')


class _PriorViewHelper(PriorViewTransformMixin):
    """Concrete helper so non-class call sites can reuse the mixin implementation."""


_VIEW_HELPER = _PriorViewHelper()


def build_router_views(
        image: Image.Image,
        view_mode: str,
        view_size: int,
        *,
        phase: str = 'inference') -> torch.Tensor:
    return _VIEW_HELPER.build_router_views(
        image,
        view_mode,
        view_size,
        phase=phase)


def process_image_for_lde_train(image, patch_size=256):
    return build_router_views(image, view_mode='dual', view_size=patch_size, phase='train')


def process_image_for_lde_inference(image, patch_size=256):
    return build_router_views(
        image,
        view_mode='dual',
        view_size=patch_size,
        phase='inference')
