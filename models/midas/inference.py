"""MiDaS depth prediction.

Pipeline::

    RGB image (PIL / HWC array / [3, H, W] tensor)
        -> built-in ImageTransform or an official MiDaS transform
        -> [1, 3, H', W'] tensor on the resolved device
        -> MiDaS model
        -> raw inverse-relative depth map [B, H', W'] (larger value = closer)
        -> bilinear resize back to the source resolution
        -> [H, W] float32 relative depth

The raw output is *relative inverse depth*: larger values are closer to the
camera and there is no metric scale (``metric_scale`` stays ``None``). Values
are never rescaled into meters here; KITTI alignment is a later project step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from models.midas.model import DEFAULT_VARIANT, MiDaSModel, MiDaSError, build_midas_model
from preprocessing.transforms import DEFAULT_MEAN, DEFAULT_STD, ImageTransform
from utils.config import get
from utils.device import cuda_available, resolve_device

LOGGER = logging.getLogger(__name__)


def _to_pil(
    image: Any, image_size: Sequence[int] | None = None
) -> tuple[Image.Image, tuple[int, int]]:
    """Normalize an accepted image form to a PIL RGB image plus its (W, H) size.

    Accepted forms (mirroring the U-Net inference wrapper): PIL image, HWC RGB
    array, or a ``[3, H, W]`` float tensor in [0, 1].
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB"), image.size

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise MiDaSError(f"Expected an HWC RGB array, got shape {image.shape}")
        source = (image.shape[1], image.shape[0])
        if image.dtype.kind == "f" and image.max() <= 1.0:
            image = (image * 255.0)
        array = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB"), source

    if isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise MiDaSError(
                f"Expected a [3, H, W] image tensor, got {image.ndim} dims; "
                f"use predict_batch for [B, 3, H, W] tensors."
            )
        array = image.detach().cpu().permute(1, 2, 0).numpy()
        source = (array.shape[1], array.shape[0])
        array = np.clip(array, 0.0, 1.0)
        return Image.fromarray((array * 255).astype(np.uint8), mode="RGB"), source

    raise MiDaSError(
        f"Unsupported image type {type(image).__name__}; expected PIL.Image, "
        f"an HWC RGB array, or a [3, H, W] tensor."
    )


def _as_batch_2d(output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
    """Coerce a model output into a ``[B, H', W']`` float tensor."""
    if output.ndim == 4:
        if output.shape[1] == 1:
            output = output[:, 0]
        else:
            raise MiDaSError(
                f"Expected a single-channel depth map, got {output.shape[1]} channels."
            )
    if output.ndim != 3:
        raise MiDaSError(
            f"Unexpected depth output shape {tuple(output.shape)}; expected [B, H, W]."
        )
    if output.shape[0] != model_input.shape[0]:
        raise MiDaSError(
            f"Depth batch size {output.shape[0]} does not match input batch "
            f"size {model_input.shape[0]}."
        )
    if output.shape[-2:] != tuple(model_input.shape[-2:]):
        raise MiDaSError(
            f"Depth output {tuple(output.shape[-2:])} does not match the model "
            f"input spatial size {tuple(model_input.shape[-2:])}."
        )
    return output.float()


def _resize_bilinear(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a ``[B, H, W]`` depth tensor to a PIL ``(W, H)`` size."""
    target_h, target_w = int(size[1]), int(size[0])
    if tensor.shape[-2:] == (target_h, target_w):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze(1)


class MidDepthPredictor:
    """Run MiDaS inference on a single image or a batch of tensors.

    Args:
        model: A :class:`MiDaSModel` or any compatible ``torch.nn.Module``
            backend (stub models are fine for tests). Constructing a wrapper
            never downloads weights.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or a ``torch.device``.
            Explicit CUDA requests on a machine without CUDA raise
            :class:`MiDaSError`.
        input_size: Optional square ``[H, W]`` resize used by the built-in
            transform (e.g. 384, the MiDaS training resolution). ``None`` keeps
            the source resolution. Final predictions are always resized back to
            the source resolution with bilinear interpolation.
        transform: Optional official MiDaS transform callable (e.g.
            ``hub_transforms.dpt_transform``). It receives an HWC uint8 RGB
            array and returns a ``[1, 3, H, W]`` tensor; when provided it
            replaces the built-in transform.
        use_official_transform: Records whether an official transform is in
            use (informational; the real selection is ``transform is not None``).
        normalize: Apply ImageNet normalization in the built-in transform.
    """

    depth_representation = "relative"
    metric_scale = None
    raw_output_larger_is_closer = True

    def __init__(
        self,
        model: nn.Module | None = None,
        device: str | torch.device = "auto",
        input_size: int | None = None,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
        use_official_transform: bool = False,
        normalize: bool = True,
        variant: str = DEFAULT_VARIANT,
    ) -> None:
        if isinstance(device, torch.device):
            device_obj = device
        else:
            device_obj = resolve_device(device)
        if "cuda" in str(device_obj) and not cuda_available():
            raise MiDaSError(f"CUDA requested but not available: {device}")

        self.device = device_obj
        self.input_size = int(input_size) if input_size is not None else None
        self.use_official_transform = bool(use_official_transform)
        self.transform = transform
        self.variant = variant

        if isinstance(model, MiDaSModel):
            self.model = model
        elif isinstance(model, nn.Module):
            self.model = MiDaSModel(backend=model, variant=variant)
        elif model is None:
            raise MiDaSError("Provide a MiDaSModel or an nn.Module backend to predict.")
        else:
            raise MiDaSError(
                f"model must be a MiDaSModel or torch.nn.Module, got {type(model).__name__}."
            )

        self.model.to(self.device)
        self.model.eval()

        builtin_size = (self.input_size, self.input_size) if self.input_size else None
        self._builtin = ImageTransform(size=builtin_size, mean=DEFAULT_MEAN, std=DEFAULT_STD, normalize=normalize)

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        model: nn.Module | None = None,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> "MidDepthPredictor":
        """Build a predictor from a YAML configuration dictionary.

        ``model`` should be provided for offline/test use; when omitted a real
        pretrained MiDaS model is built through :func:`build_midas_model`,
        which may download or hit the hub cache.
        """
        model_cfg = get(cfg, "model", {}) or {}
        pre_cfg = get(cfg, "preprocessing", {}) or {}
        inf_cfg = get(cfg, "inference", {}) or {}
        if model is None:
            model = build_midas_model(
                variant=get(model_cfg, "variant", DEFAULT_VARIANT),
                weights_path=get(model_cfg, "weights_path"),
                device=get(inf_cfg, "device", "auto"),
            )
        return cls(
            model=model,
            device=get(inf_cfg, "device", "auto"),
            input_size=get(pre_cfg, "input_size"),
            use_official_transform=get(pre_cfg, "use_official_transform", False),
            transform=transform,
        )

    def preprocess(self, image: Any) -> tuple[torch.Tensor, tuple[int, int]]:
        """Convert an image into a ``[1, 3, H', W']`` batch and its source size."""
        pil, source_size = _to_pil(image)
        if self.transform is not None:
            array = np.asarray(pil, dtype=np.uint8)
            batch = self.transform(array)
            if not isinstance(batch, torch.Tensor):
                raise MiDaSError(
                    f"Transform returned {type(batch).__name__}; expected a torch.Tensor."
                )
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            if batch.dim() != 4 or batch.shape[1] != 3:
                raise MiDaSError(
                    f"Transform returned shape {tuple(batch.shape)}; expected [1, 3, H, W]."
                )
            return batch.to(self.device), source_size
        return self._builtin(pil).unsqueeze(0).to(self.device), source_size

    def predict(self, image: Any) -> torch.Tensor:
        """Predict a ``[H, W]`` float32 depth map at the source resolution."""
        batch, source_size = self.preprocess(image)
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch.to(self.device))
            depth = _as_batch_2d(output, model_input=batch)
            depth = _resize_bilinear(depth, source_size)
        return depth.squeeze(0).cpu()

    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Predict for a ``[B, 3, H, W]`` tensor assumed pre-processed.

        The tensor is expected to follow the model input convention already
        (normalized, at the model resolution); use :meth:`preprocess` per image
        when automatic resize-back is required. Returns a ``[B, H, W]`` float32
        tensor.
        """
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise MiDaSError(
                f"predict_batch expects a [B, 3, H, W] tensor, got: "
                f"{type(images).__name__} {tuple(images.shape) if hasattr(images, 'shape') else ''}"
            )
        self.model.eval()
        with torch.no_grad():
            output = self.model(images.to(self.device))
            depth = _as_batch_2d(output, model_input=images)
        return depth.cpu()

    def normalize_visualization(self, depth: torch.Tensor) -> torch.Tensor:
        """Map a raw (inverse) relative depth map to ``[0, 1]`` for display.

        Visualization-only: the result inverts the inverse-depth convention so
        that larger values mean *farther* from the camera (matching
        ``output.larger_is_farther`` in the config). This is NOT metric depth
        calibration and never touches the raw prediction.
        """
        d = depth.float()
        vmin, vmax = d.min(), d.max()
        if not (vmax > vmin):
            return torch.zeros_like(d)
        d = (d - vmin) / (vmax - vmin)
        return 1.0 - d