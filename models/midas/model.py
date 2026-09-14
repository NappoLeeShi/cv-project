"""MiDaS monocular depth model wrapper.

Convention (from the official MiDaS / PyTorch Hub documentation): MiDaS predicts
*relative inverse depth* (disparity-like). The raw output has no metric scale —
``is_metric`` is ``False`` and ``metric_scale`` stays ``None``; converting to
meters is a separate evaluation step (later in the project). By default the
*raw* value is larger for pixels closer to the camera.

Model construction and weight loading are kept separate: building a
:class:`MiDaSModel` never downloads anything; :func:`build_midas_model` and
:func:`build_midas_transform` are the only entry points that may contact the
torch.hub source or read a checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from utils.device import resolve_device

LOGGER = logging.getLogger(__name__)

MODEL_SOURCE = "intel-isl/MiDaS"
SUPPORTED_VARIANTS = ("dpt_large", "dpt_hybrid", "midas_small")
DEFAULT_VARIANT = "dpt_hybrid"


class MiDaSError(RuntimeError):
    """Raised when a MiDaS model cannot be built, loaded, or used."""


def validate_variant(variant: str) -> str:
    lower = str(variant).lower()
    if lower not in SUPPORTED_VARIANTS:
        raise MiDaSError(f"Unsupported MiDaS variant {variant!r}; choose from {SUPPORTED_VARIANTS}")
    return lower


class MiDaSModel(nn.Module):
    """Thin wrapper around a MiDaS-compatible backend.

    Attributes:
        name: Always ``"midas"`` (model type).
        variant: e.g. ``"dpt_hybrid"``.
        raw_output_is_inverse_relative: ``True``; larger raw value = closer.
        is_metric: ``False``; the output is NOT in meters.
    """

    raw_output_is_inverse_relative = True
    is_metric = False

    def __init__(self, backend: nn.Module, variant: str = DEFAULT_VARIANT, name: str = "midas") -> None:
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise MiDaSError(
                f"backend must be a torch.nn.Module, got {type(backend).__name__}"
            )
        self.name = name
        self.variant = validate_variant(variant)
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend(x)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "variant": self.variant,
            "source": MODEL_SOURCE,
            "representation": "inverse relative depth",
            "larger_is_closer": True,
            "is_metric": self.is_metric,
            "metric_scale": None,
        }


def _extract_state_dict(checkpoint: object, source: str) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        state = checkpoint
    else:
        raise MiDaSError(
            f"Unrecognized weights format in {source}: expected a state_dict or a "
            f"dict containing a 'model_state_dict' key."
        )
    if not isinstance(state, dict):
        raise MiDaSError(f"Weights {source} do not contain a state_dict mapping.")
    return state


def _load_weights(backend: nn.Module, weights_path: str | Path, device: torch.device) -> None:
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise MiDaSError(f"Weights file not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=backend.device, weights_only=True)
    except Exception as exc:
        raise MiDaSError(f"Failed to load MiDaS weights from {path}: {exc}") from exc
    state = _extract_state_dict(checkpoint, str(path))
    model_state = backend.state_dict()
    missing = [k for k in model_state if k not in state]
    unexpected = [k for k in state if k not in model_state]
    mismatched = [
        k
        for k in model_state
        if k in state and tuple(state[k].shape) != tuple(model_state[k].shape)
    ]
    if missing or unexpected or mismatched:
        raise MiDaSError(
            f"MiDaS weights {path} are incompatible with the {type(backend).__name__} "
            f"architecture (missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"mismatched={mismatched[:8]})."
        )
    backend.load_state_dict(state, strict=True)


def build_midas_model(
    variant: str = DEFAULT_VARIANT,
    weights_path: str | Path | None = None,
    device: str | torch.device = "auto",
) -> MiDaSModel:
    """Build a MiDaS backend via torch.hub and wrap it.

    Without ``weights_path`` the pretrained hub weights are used (this may
    download or hit the hub cache). With ``weights_path`` the architecture is
    built unpretrained and the given checkpoint is loaded instead. A clear
    :class:`MiDaSError` is raised whenever the source/weights are unavailable.
    """
    variant = validate_variant(variant)
    device_obj = resolve_device(device) if not isinstance(device, torch.device) else device
    try:
        backend = torch.hub.load(
            MODEL_SOURCE, variant, pretrained=(weights_path is None)
        )
    except Exception as exc:
        raise MiDaSError(
            f"Could not load MiDaS {variant!r} from torch.hub ({MODEL_SOURCE}). "
            f"Ensure the weights are cached or a local {weights_path!r} is provided. "
            f"Inner error: {exc}"
        ) from exc
    if weights_path is not None:
        _load_weights(backend, weights_path, device_obj)
    model = MiDaSModel(backend=backend, variant=variant).to(device_obj)
    model.eval()
    return model


def build_midas_transform(variant: str = DEFAULT_VARIANT):
    """Load the official MiDaS transform set from torch.hub.

    Returns an object exposing ``dpt_transform`` / ``small_transform`` for the
    chosen variant. May require network access or a populated hub cache (any
    failure is reported as a :class:`MiDaSError`).
    """
    variant = validate_variant(variant)
    try:
        transforms = torch.hub.load(MODEL_SOURCE, "transforms")
    except Exception as exc:
        raise MiDaSError(
            f"Could not load MiDaS transforms from torch.hub ({MODEL_SOURCE}): {exc}"
        ) from exc
    return transforms