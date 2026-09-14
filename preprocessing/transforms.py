"""Shared preprocessing helpers used by the dataset loaders.

The design keeps the heavy lifting (label remapping, depth scaling) inside the
dataset modules and centralizes only the tensor conversions and resize rules:
* Images are resized bilinearly and normalised.
* Segmentation labels are resized with nearest-neighbour interpolation only.
* Depth maps are resized bilinearly and the valid mask is recomputed from the
  resized depth itself, so resized values never leak into invalid pixels.

Size conventions: config values are [H, W]; PIL ``resize`` takes (W, H), which
is handled by :func:`hw_to_pil_size`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

_DEFAULT_SIZE = (512, 1024)


def hw_to_pil_size(size: Sequence[int]) -> tuple[int, int]:
    """Convert a config-style ``[H, W]`` size to a PIL ``(W, H)`` tuple."""
    h, w = int(size[0]), int(size[1])
    return w, h


def resize_nearest(image: Image.Image, size: Sequence[int]) -> Image.Image:
    """Resize a label/single-channel image with nearest-neighbour sampling."""
    if size is None:
        return image
    return image.resize(hw_to_pil_size(size), Image.Resampling.NEAREST)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a ``torch.Tensor`` of shape ``(3, H, W)``."""
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0


def label_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert an integer label array to a ``torch.long`` tensor."""
    return torch.as_tensor(array, dtype=torch.long)


class ImageTransform:
    """Resize, tensorize and optionally normalise an RGB image.

    Args:
        size: Target ``[H, W]``. ``None`` keeps the source resolution.
        mean / std: Per-channel normalisation constants.
        normalize: Apply ``(x - mean) / std`` when ``True``.
    """

    def __init__(
        self,
        size: Sequence[int] | None = None,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        normalize: bool = True,
    ) -> None:
        self.size = tuple(size) if size is not None else None
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.normalize = normalize

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.size is not None and image.size != hw_to_pil_size(self.size):
            image = image.resize(hw_to_pil_size(self.size), Image.Resampling.BILINEAR)
        tensor = image_to_tensor(image)
        if self.normalize:
            mean = torch.as_tensor(self.mean).view(3, 1, 1)
            std = torch.as_tensor(self.std).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor


class DepthTransform:
    """Resize a depth map and recompute its valid mask.

    Depth is already in metric units when it reaches this transform. Resizing is
    bilinear; the returned mask equals ``depth > 0`` after resizing, so an
    invalid pixel stays ``0.0`` and gets masked out regardless of interpolation.
    """

    def __init__(self, size: Sequence[int] | None = None) -> None:
        self.size = tuple(size) if size is not None else None

    def __call__(self, depth: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.fromarray(depth, mode="F")
        if self.size is not None and image.size != hw_to_pil_size(self.size):
            image = image.resize(hw_to_pil_size(self.size), Image.Resampling.BILINEAR)
        resized = np.asarray(image, dtype=np.float32).copy()
        if resized.ndim == 3:
            resized = resized[..., 0]
        dist = torch.from_numpy(resized)
        valid = dist > 0
        depth_tensor = torch.where(valid, dist, torch.zeros_like(dist))
        return depth_tensor, valid