"""Semantic segmentation visualization.

Maps Cityscapes trainId class IDs (0..18, 255 = ignore) to a fixed,
deterministic RGB palette and renders overlays on the original image.
Visualization only reads model/analysis outputs; it never changes them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from scene_understanding.fusion import CITYSCAPES_TRAINID_NAMES

# Fixed 19-class Cityscapes palette (trainId order 0..18). Deterministic.
CITYSCAPES_TRAINID_COLORS: tuple[tuple[int, int, int], ...] = (
    (128, 64, 128),   # 0  road
    (244, 35, 232),   # 1  sidewalk
    (70, 70, 70),     # 2  building
    (102, 102, 156),  # 3  wall
    (190, 153, 153),  # 4  fence
    (153, 153, 153),  # 5  pole
    (250, 170, 30),   # 6  traffic light
    (220, 220, 0),    # 7  traffic sign
    (107, 142, 35),   # 8  vegetation
    (152, 251, 152),  # 9  terrain
    (70, 130, 180),   # 10 sky
    (220, 20, 60),    # 11 person
    (255, 0, 0),      # 12 rider
    (0, 0, 142),      # 13 car
    (0, 0, 70),       # 14 truck
    (0, 60, 100),     # 15 bus
    (0, 80, 100),     # 16 train
    (0, 0, 230),      # 17 motorcycle
    (119, 11, 32),    # 18 bicycle
)
VOID_COLOR: tuple[int, int, int] = (0, 0, 0)


def _validate_segmentation(segmentation) -> np.ndarray:
    array = np.asarray(segmentation)
    if array.ndim != 2:
        raise ValueError(
            f"segmentation map must be 2D [H, W], got {array.ndim} dim(s)"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"segmentation map must contain integer train IDs, got dtype {array.dtype}"
        )
    return array


def _validate_rgb_image(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"image must be RGB [H, W, 3], got shape {array.shape}"
        )
    return array


def colorize_segmentation(
    segmentation: np.ndarray,
    palette: Sequence[Sequence[int]] | None = None,
    ignore_index: int = 255,
) -> np.ndarray:
    """Map a semantic class-ID map to a fixed RGB image [H, W, 3] (uint8).

    Args:
        segmentation: ``[H, W]`` integer Cityscapes trainId map.
        palette: Sequence of ``(R, G, B)`` triplets; defaults to the fixed
            19-class Cityscapes palette.
        ignore_index: Class ID rendered as black (default 255 = void).

    Returns:
        ``uint8`` array of shape ``[H, W, 3]``.
    """
    seg = _validate_segmentation(segmentation)
    if palette is None:
        colors = CITYSCAPES_TRAINID_COLORS
    else:
        colors = tuple(tuple(int(c) for c in entry) for entry in palette)
    if not colors:
        raise ValueError("palette must contain at least one color")
    lookup = np.asarray(colors, dtype=np.uint8)
    if lookup.ndim != 2 or lookup.shape[1] != 3:
        raise ValueError("palette must be a sequence of (R, G, B) triplets")

    flat = seg.reshape(-1)
    present = np.unique(flat)
    valid = present[present != ignore_index]
    if valid.size:
        low, high = int(valid.min()), int(valid.max())
        if low < 0:
            raise ValueError(f"segmentation map contains negative class ID {low}")
        if high >= len(colors):
            raise ValueError(
                f"segmentation map contains class ID {high} beyond a "
                f"{len(colors)}-entry palette (and not the ignore index {ignore_index})"
            )

    colored = np.zeros((flat.size, 3), dtype=np.uint8)
    mapped = (flat != ignore_index) & (flat >= 0) & (flat < len(colors))
    colored[mapped] = np.take(lookup, flat[mapped].astype(np.intp), axis=0)
    colored[~mapped] = VOID_COLOR
    return colored.reshape((seg.shape[0], seg.shape[1], 3))


def create_segmentation_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend a colored segmentation onto the original RGB image.

    Args:
        image: RGB image ``[H, W, 3]`` (uint8 or float in ``[0, 1]``).
        segmentation: ``[H, W]`` integer trainId map (not modified).
        alpha: Overlay weight in ``[0, 1]``.

    Returns:
        ``uint8`` ``[H, W, 3]`` overlay.
    """
    img = _validate_rgb_image(image)
    seg = _validate_segmentation(segmentation)
    if img.shape[:2] != seg.shape:
        raise ValueError(
            "image and segmentation must share spatial dimensions, "
            f"got {img.shape[:2]} vs {seg.shape}"
        )
    weight = float(alpha)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if img.dtype == np.uint8:
        base = img
    elif img.dtype.kind == "f":
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError(
                "float images are expected in [0, 1]; use uint8 for 0..255 values"
            )
        base = (img * 255.0).round().astype(np.uint8)
    else:
        raise ValueError(f"unsupported image dtype {img.dtype}; use uint8 or float")

    seg_rgb = colorize_segmentation(seg)
    blended = (base.astype(np.float32) * (1.0 - weight)
               + seg_rgb.astype(np.float32) * weight)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def create_segmentation_legend() -> Figure:
    """Return a matplotlib Figure with the 19-class legend (name / ID / color)."""
    fig = Figure(figsize=(6.5, 0.42 * len(CITYSCAPES_TRAINID_COLORS)))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_axis_off()
    classes = sorted(CITYSCAPES_TRAINID_NAMES)
    step = 1.0 / max(len(classes), 1)
    for i, class_id in enumerate(classes):
        color = CITYSCAPES_TRAINID_COLORS[class_id]
        y = 1.0 - (i + 0.5) * step
        ax.add_patch(
            Rectangle((0.0, y - 0.28 * step), 0.55 * step, 0.56 * step,
                      facecolor=np.asarray(color) / 255.0, edgecolor="none")
        )
        ax.text(
            0.45 * step, y, f"{class_id}  {CITYSCAPES_TRAINID_NAMES[class_id]}",
            ha="left", va="center", fontsize=10,
        )
    ax.set_xlim(0, max(1.0, 0.45 * step + 1.0))
    ax.set_ylim(0, 1)
    return fig


def save_segmentation(prediction, output_path) -> None:
    """Legacy helper: save a class-ID prediction as a grayscale PNG.

    Kept for backward compatibility with ``main.py``; the Step 10 API is
    :func:`colorize_segmentation` + :func:`save_visualization`.
    """
    prediction = prediction.numpy()
    image = Image.fromarray(np.uint8(prediction))
    image.save(output_path)