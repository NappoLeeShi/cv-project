"""U-Net inference for semantic segmentation.

Pipeline::

    RGB image (PIL / HWC array / tensor)
        -> preprocessing.ImageTransform (single normalization source)
        -> [1, 3, H, W] tensor on the resolved device
        -> U-Net logits [1, C, H, W]
        -> argmax(dim=1) -> integer class map
        -> (optional) nearest-neighbour resize back to the source resolution

Predictions stay in Cityscapes trainId space (0..18). The model output is
resized back to the source resolution with nearest-neighbour interpolation
only; bilinear interpolation is never used for class IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from models.unet.model import UNet
from preprocessing.transforms import DEFAULT_MEAN, DEFAULT_STD, ImageTransform
from utils.config import get
from utils.device import cuda_available, resolve_device

_UNetModel = nn.Module


class InferenceError(RuntimeError):
    """Raised when a model, checkpoint, or image cannot be used for inference."""


def _extract_state_dict(checkpoint: Any, source: str) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        state = checkpoint
    else:
        raise InferenceError(
            f"Unrecognized checkpoint format in {source}: expected either a raw "
            f"state_dict or a dict containing a 'model_state_dict' key."
        )
    if not isinstance(state, dict):
        raise InferenceError(f"Checkpoint {source} does not contain a state_dict mapping.")
    return state


def _apply_state_dict(model: _UNetModel, state: dict[str, torch.Tensor], source: str) -> None:
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    mismatched = [
        key
        for key in model_state
        if key in state and tuple(state[key].shape) != tuple(model_state[key].shape)
    ]
    if missing or unexpected or mismatched:
        error = f"Checkpoint {source} is incompatible with the model"
        if missing:
            error += f"; missing keys: {missing[:8]}"
        if unexpected:
            error += f"; unexpected keys: {unexpected[:8]}"
        if mismatched:
            error += f"; shape-mismatched keys: {mismatched[:8]}"
        raise InferenceError(error)
    model.load_state_dict(state, strict=True)


def load_model(model: _UNetModel, weight_path: str | Path | None = None, device: str = "cpu") -> _UNetModel:
    """Load a U-Net checkpoint onto ``device`` and place the model in eval mode.

    Supports both ``torch.save(model.state_dict(), path)`` and
    ``torch.save({"model_state_dict": ...}, path)``. Incompatible checkpoints
    raise a clear :class:`InferenceError`.
    """
    device = resolve_device(device)
    model = model.to(device)
    if weight_path is not None:
        path = Path(weight_path).expanduser()
        if not path.is_file():
            raise InferenceError(f"Checkpoint file not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except Exception as exc:
            raise InferenceError(f"Failed to load checkpoint {path}: {exc}") from exc
        state = _extract_state_dict(checkpoint, str(path))
        _apply_state_dict(model, state, str(path))
    model.eval()
    return model


def _to_pil(image: Any) -> tuple[Image.Image, tuple[int, int]]:
    if isinstance(image, Image.Image):
        return image.convert("RGB"), image.size

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise InferenceError(f"Expected an HWC RGB array, got shape {image.shape}")
        return _from_array(image, (image.shape[1], image.shape[0]))

    if isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise InferenceError(
                f"Expected a [3, H, W] image tensor, got {image.ndim} dims; "
                f"use predict_batch for [B, 3, H, W] tensors."
            )
        array = image.detach().cpu().permute(1, 2, 0).numpy()
        return _from_array(array, (array.shape[1], array.shape[0]))

    raise InferenceError(
        f"Unsupported image type {type(image).__name__}; expected PIL.Image, "
        f"an HWC RGB array, or a [3, H, W] tensor."
    )


def _from_array(array: np.ndarray, source_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    if array.dtype.kind == "f":
        if array.max() <= 1.0:
            array = (array * 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB"), source_size


def _resize_nearest_class_ids(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = int(size[1]), int(size[0])
    if tensor.shape[-2:] == (h, w):
        return tensor
    four = tensor.unsqueeze(1)
    return F.interpolate(four.float(), size=(h, w), mode="nearest-exact").long().squeeze(1)


def _resize_bilinear(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = int(size[1]), int(size[0])
    if tensor.shape[-2:] == (h, w):
        return tensor
    four = tensor.unsqueeze(1)
    return F.interpolate(four, size=(h, w), mode="bilinear", align_corners=False).squeeze(1)


class UNetInference:
    """Run U-Net inference on a single image or a batch of tensors.

    Args:
        model: A U-Net instance. When ``None`` a randomly initialised ``UNet``
            is built from ``num_classes`` (useful for tests / no-checkpoint runs).
        device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``) or a
            ``torch.device``. Explicit CUDA on a machine without CUDA raises.
        image_size: ``[H, W]`` the RGB image is resized to before inference.
            ``None`` keeps the source resolution. Predictions are always
            resized back to the source resolution (nearest neighbour).
        mean / std / normalize: Follow the project normalization convention
            (defaults to ImageNet stats).
        num_classes: Used only when ``model`` is ``None``.
        checkpoint: Optional path to a compatible U-Net checkpoint.
    """

    def __init__(
        self,
        model: _UNetModel | None = None,
        device: str | torch.device = "auto",
        image_size: Sequence[int] | None = None,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        normalize: bool = True,
        num_classes: int | None = None,
        checkpoint: str | Path | None = None,
    ) -> None:
        if isinstance(device, torch.device):
            device_obj = device
        else:
            device_obj = resolve_device(device)
        if "cuda" in str(device_obj) and not cuda_available():
            raise InferenceError(f"CUDA requested but not available: {device}")

        if model is None:
            if num_classes is None:
                raise InferenceError("Provide a model or a num_classes to build one.")
            model = UNet(num_classes=num_classes)

        self.model: _UNetModel = model.to(device_obj)
        self.device = device_obj
        self.image_size = tuple(image_size) if image_size is not None else None
        self.transform = ImageTransform(size=self.image_size, mean=mean, std=std, normalize=normalize)

        if checkpoint is not None:
            load_model(self.model, weight_path=checkpoint, device=str(device_obj))
        self.model.eval()

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        model: _UNetModel | None = None,
        checkpoint: str | Path | None = None,
    ) -> "UNetInference":
        """Build an inference wrapper from a YAML configuration dictionary."""
        inference_cfg = get(cfg, "inference", {}) or {}
        model_cfg = get(cfg, "model", {}) or {}
        data_cfg = get(cfg, "data", {}) or {}
        return cls(
            model=model,
            device=get(inference_cfg, "device", "auto"),
            image_size=get(inference_cfg, "image_size") or get(data_cfg, "image_size"),
            mean=get(data_cfg, "normalization.mean") or DEFAULT_MEAN,
            std=get(data_cfg, "normalization.std") or DEFAULT_STD,
            normalize=get(inference_cfg, "normalize", True),
            num_classes=get(model_cfg, "num_classes"),
            checkpoint=checkpoint,
        )

    def preprocess(self, image: Any) -> torch.Tensor:
        """Apply the project transforms and return a ``[1, 3, H', W']`` batch."""
        pil, _ = _to_pil(image)
        return self.transform(pil).unsqueeze(0).to(self.device)

    def predict(self, image: Any, return_confidence: bool = False) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Predict the class map for one image.

        Args:
            image: PIL image, HWC RGB array, or ``[3, H, W]`` tensor.
            return_confidence: Also return a per-pixel confidence map of the
                same resolution, where confidence = max softmax probability.

        Returns:
            A ``[H, W]`` ``torch.long`` class map (trainId 0..18), or a
            ``(prediction, confidence)`` pair when ``return_confidence`` is set.
        """
        pil, source_size = _to_pil(image)
        batch = self.preprocess(pil)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)
            prediction = logits.argmax(dim=1)
            if self.image_size is not None:
                prediction = _resize_nearest_class_ids(prediction, source_size)
            prediction = prediction.squeeze(0).cpu()

        if not return_confidence:
            return prediction

        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities.max(dim=1).values
        if self.image_size is not None:
            confidence = _resize_bilinear(confidence, source_size)
        confidence = confidence.squeeze(0).cpu()
        return prediction, confidence

    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Predict class maps for a ``[B, 3, H, W]`` tensor.

        The tensor is assumed to already follow the model input convention
        (normalised, at the model resolution); use the dataset/preprocessing
        modules to produce it. Returns a ``[B, H, W]`` ``torch.long`` tensor.
        """
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise InferenceError(
                f"predict_batch expects a [B, 3, H, W] tensor, got: "
                f"{type(images).__name__} {tuple(images.shape) if hasattr(images, 'shape') else ''}"
            )
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images.to(self.device))
            prediction = logits.argmax(dim=1)
        return prediction.cpu()