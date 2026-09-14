"""Fusion visualization: segmentation over relative depth in one image.

Keeps the MiDaS convention (larger inverse depth = closer) and never
represents depth as meters. Produces RGB ``uint8`` arrays by combining the
semantic overlay with a depth colormap; no neural fusion model involved.
"""

from __future__ import annotations

import numpy as np

from scene_understanding.fusion import (
    REGION_FAR,
    REGION_MIDDLE,
    REGION_NEAR,
)
from visualization.depth import colorize_depth, normalize_depth_for_visualization
from visualization.segmentation import (
    VOID_COLOR,
    _validate_rgb_image,
    _validate_segmentation,
    create_segmentation_overlay,
)

# Deterministic region colors: near (warm, closer) -> far (cool).
REGION_COLORS: dict[int, tuple[int, int, int]] = {
    REGION_NEAR: (255, 60, 60),
    REGION_MIDDLE: (255, 200, 0),
    REGION_FAR: (60, 140, 255),
}
REGION_INVALID: int = -1


def create_fusion_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    depth: np.ndarray,
    alpha_seg: float = 0.5,
    alpha_depth: float = 0.35,
) -> np.ndarray:
    """Blend the semantic overlay and the depth colormap onto the RGB image.

    Args:
        image: RGB image ``[H, W, 3]``.
        segmentation: ``[H, W]`` integer trainId map (not modified).
        depth: ``[H, W]`` relative inverse depth (not modified).
        alpha_seg: Weight of the segmentation coloring in ``[0, 1]``.
        alpha_depth: Weight of the depth coloring in ``[0, 1]``.

    Returns:
        ``uint8`` ``[H, W, 3]`` fused visualization.
    """
    img = _validate_rgb_image(image)
    depth_arr = np.asarray(depth)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth map must be 2D [H, W], got {depth_arr.ndim} dim(s)")
    if depth_arr.shape != img.shape[:2]:
        raise ValueError(
            "image and depth must share spatial dimensions, "
            f"got {img.shape[:2]} vs {depth_arr.shape}"
        )
    seg_weight = float(alpha_seg)
    if not 0.0 <= seg_weight <= 1.0:
        raise ValueError(f"alpha_seg must be in [0, 1], got {alpha_seg}")
    depth_weight = float(alpha_depth)
    if not 0.0 <= depth_weight <= 1.0:
        raise ValueError(f"alpha_depth must be in [0, 1], got {alpha_depth}")

    seg_overlay = create_segmentation_overlay(image, segmentation, alpha=seg_weight)
    depth_rgb = colorize_depth(depth_arr)
    blended = (seg_overlay.astype(np.float32) * (1.0 - depth_weight)
               + depth_rgb.astype(np.float32) * depth_weight)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def create_region_visualization(
    segmentation: np.ndarray,
    depth: np.ndarray,
    region_map: np.ndarray,
    region_colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render the near/middle/far region map with depth structure.

    Region codes follow :mod:`scene_understanding.fusion` (``0`` far, ``1``
    middle, ``2`` near, ``-1`` not analyzed). Regions get fixed colors and are
    shaded by the (normalized) relative depth so near pixels read brighter.

    Returns:
        ``uint8`` ``[H, W, 3]`` region visualization.
    """
    seg = _validate_segmentation(segmentation)
    depth_arr = _validate_depth_for_regions(depth)
    regions = np.asarray(region_map)
    if regions.ndim != 2:
        raise ValueError(
            f"region map must be 2D [H, W], got {regions.ndim} dim(s)"
        )
    if not (regions.shape == seg.shape == depth_arr.shape):
        raise ValueError(
            "segmentation, depth and region map must share spatial dimensions, "
            f"got {seg.shape}, {depth_arr.shape}, {regions.shape}"
        )
    accepted = (REGION_INVALID, REGION_FAR, REGION_MIDDLE, REGION_NEAR)
    unknown = set(np.unique(regions)) - set(accepted)
    if unknown:
        raise ValueError(
            f"region map contains unknown code(s) {sorted(unknown)}; "
            "expected near=2 / middle=1 / far=0 / invalid=-1"
        )

    colors = dict(REGION_COLORS if region_colors is None else region_colors)
    for expected in (REGION_FAR, REGION_MIDDLE, REGION_NEAR):
        if expected not in colors:
            raise ValueError(f"region_colors must define region {expected}")

    base = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for code in (REGION_FAR, REGION_MIDDLE, REGION_NEAR):
        mask = regions == code
        base[mask] = colors[code]
    base[regions == REGION_INVALID] = VOID_COLOR

    depth_norm = normalize_depth_for_visualization(depth_arr)
    depth_norm[np.asarray(regions == REGION_INVALID)] = 0.0
    brightness = 0.55 + 0.45 * depth_norm
    shaded = (base.astype(np.float32) * brightness[..., np.newaxis])
    return np.clip(np.rint(shaded), 0, 255).astype(np.uint8)


def _validate_depth_for_regions(depth) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim != 2:
        raise ValueError(f"depth map must be 2D [H, W], got {arr.ndim} dim(s)")
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"depth map must be numeric, got dtype {arr.dtype}")
    return arr.astype(np.float64)