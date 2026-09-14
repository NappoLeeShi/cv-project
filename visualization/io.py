"""Save helpers for visualization outputs.

Functions stay side-effect-free unless the caller explicitly requests a save.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

SUPPORTED_RASTER_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


def _validate_output_path(output_path) -> tuple[Path, str]:
    path = Path(os.fspath(output_path))
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_RASTER_EXTENSIONS:
        raise ValueError(
            f"unsupported output extension {suffix!r}; "
            f"use one of {SUPPORTED_RASTER_EXTENSIONS}"
        )
    return path, suffix


def _as_uint8_rgb(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., np.newaxis], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"image must be [H, W] or [H, W, 3], got shape {array.shape}")
    if array.dtype != np.uint8:
        if array.dtype.kind == "f":
            array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return array[..., :3]


def save_visualization(image, output_path) -> str:
    """Save an RGB visualization image to a file (parent dirs auto-created).

    Supports ``.png`` / ``.jpg`` / ``.jpeg``. Converts the array to ``uint8``
    RGB on a copy; the source object is never modified.
    """
    path, _ = _validate_output_path(output_path)
    rgb = _as_uint8_rgb(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    return str(path)


def save_figure(fig: Figure, output_path, dpi: int = 150) -> str:
    """Save a matplotlib Figure to a file, creating parent directories.

    The requested extension selects the output format (``.png``/``.jpg``).
    """
    path, suffix = _validate_output_path(output_path)
    if not isinstance(fig, Figure):
        raise TypeError(f"expected a matplotlib Figure, got {type(fig).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format=suffix.lstrip("."))
    return str(path)