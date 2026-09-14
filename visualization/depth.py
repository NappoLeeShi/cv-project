"""Relative inverse depth visualization.

The MiDaS depth prediction is **relative inverse depth** (larger = closer to
the camera). This module only reads it: normalization happens on a derived
copy for display, the raw prediction is never modified, and nothing here
labels depth as metric depth or meters.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

DEFAULT_COLORMAP: str = "turbo"
COLORBAR_LABEL: str = "Relative inverse depth (larger = closer)"


def _validate_depth(depth) -> np.ndarray:
    array = np.asarray(depth)
    if array.ndim != 2:
        raise ValueError(f"depth map must be 2D [H, W], got {array.ndim} dim(s)")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"depth map must be numeric, got dtype {array.dtype}")
    if array.dtype.kind == "b":
        raise ValueError("depth map must not be boolean")
    return array


def normalize_depth_for_visualization(depth: np.ndarray) -> np.ndarray:
    """Return a float32 ``[H, W]`` copy of the depth in ``[0, 1]``.

    NaN/inf pixels are masked out; completely invalid or constant maps become
    all zeros. Safe to call on any relative-depth map.
    """
    arr = _validate_depth(depth)
    finite = np.isfinite(arr)
    values = arr.astype(np.float32)[finite]
    if values.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    normalized = np.zeros(arr.shape, dtype=np.float32)
    if vmax > vmin:
        normalized[finite] = (arr[finite].astype(np.float32) - vmin) / (vmax - vmin)
    return normalized


def _as_float32_rgb(rgba: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgba[..., :3] * 255.0), 0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray, colormap: str = DEFAULT_COLORMAP) -> np.ndarray:
    """Render a relative-depth map as an RGB ``uint8`` ``[H, W, 3]`` image.

    Larger inverse depth (closer) is encoded with a higher colormap value.
    Never modifies the input.
    """
    arr = _validate_depth(depth)
    try:
        cmap = colormaps[colormap]
    except KeyError:
        raise ValueError(f"unknown matplotlib colormap: {colormap!r}") from None
    normalized = normalize_depth_for_visualization(arr)
    return _as_float32_rgb(cmap(normalized))


def create_depth_figure(
    depth: np.ndarray,
    title: str | None = None,
    colormap: str = DEFAULT_COLORMAP,
) -> Figure:
    """Return a Figure with the depth map, a colorbar and a clear title.

    The colorbar is labelled relative inverse depth (never meters).
    """
    arr = _validate_depth(depth)
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111)
    normalized = normalize_depth_for_visualization(arr)
    im = ax.imshow(normalized, cmap=colormap, norm=Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(COLORBAR_LABEL)
    ax.set_title(title if title is not None else "Relative Inverse Depth")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig