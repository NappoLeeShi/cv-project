"""Device / environment helpers.

Small utilities to resolve the compute device once and share it across models.
Config files use ``device: auto`` to defer the choice to runtime.

"""

from __future__ import annotations

import torch


def resolve_device(device: str | None = "auto") -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    ``"auto"`` (or ``None``) selects CUDA when available, otherwise CPU.
    Any other value is passed through to :class:`torch.device`; an unknown
    device raises ``ValueError`` or ``RuntimeError`` depending on the torch
    version.
    """
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cuda_available() -> bool:
    """Whether a CUDA device is available."""
    return torch.cuda.is_available()


def device_summary() -> dict[str, object]:
    """Describe the runtime environment of the host."""
    return {
        "device": str(resolve_device("auto")),
        "cuda_available": torch.cuda.is_available(),
        "device_count": int(torch.cuda.device_count()),
        "torch_version": torch.__version__,
    }