"""Analytical (rule-based) fusion of segmentation + relative depth.

Inputs come from the SAME input RGB image through two models:
* segmentation mask ``[H, W]`` with Cityscapes trainId class IDs (0..18, 255 void)
* relative depth map ``[H, W]`` — raw MiDaS **inverse** relative depth.

Depth convention (documented and enforced consistently here):
* larger depth value  -> region closer to the camera
* smaller depth value -> region farther away
* values are relative / unit-less, never meters.

Depth regions derive their thresholds from the prediction itself (default terciles):
``near`` = top third of depth values, ``middle`` = middle third, ``far`` = bottom
third. Invalid (non-finite) depth pixels and void (255) segmentation pixels are
excluded from every statistic via an analyzed mask. Missing semantic classes
produce an absent result, never an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

VOID_ID = 255
DEPTH_REGION_NAMES = ("far", "middle", "near")
REGION_FAR, REGION_MIDDLE, REGION_NEAR = 0, 1, 2

DEFAULT_QUANTILES: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
DEFAULT_ROI_CLASSES: tuple[int, ...] = tuple(range(11, 19))  # person .. bicycle
DEFAULT_DRIVABLE_CLASSES: tuple[int, ...] = (0, 1)  # road, sidewalk

CITYSCAPES_TRAINID_NAMES: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

_CLASS_IDS_ALL = tuple(range(19))


@dataclass
class FusionResult:
    """Output of :func:`fuse`: validated inputs plus fused statistics.

    Attributes:
        seg_mask: Integer segmentation ``[H, W]`` (trainId space).
        depth: Raw **inverse** relative-depth map ``[H, W]`` (float64).
        valid_mask: ``True`` where the depth value is finite.
        analyzed_mask: ``True`` where a pixel is valid for statistics
            (finite depth AND a non-void class).
        per_class: ``class_id -> stats`` for every requested class that is
            present. Stats: ``pixel_count``, ``pixel_ratio``, ``mean_depth``,
            ``median_depth``, ``min_depth``, ``max_depth``.
        region_map: ``[H, W]`` code per pixel: ``0`` far, ``1`` middle,
            ``2`` near, ``-1`` not analyzed.
        region_summary: ``{"far","middle","near"} -> {"pixel_count",
            "pixel_ratio"}`` plus ``"thresholds": {"low", "high"}``.
        thresholds: ``(low, high)`` inverse-depth region boundaries
            (``depth < low`` far, ``depth >= high`` near).
    """

    seg_mask: np.ndarray
    depth: np.ndarray
    valid_mask: np.ndarray
    analyzed_mask: np.ndarray
    per_class: dict[int, dict]
    region_map: np.ndarray
    region_summary: dict
    thresholds: tuple[float, float]


def _as_2d_array(value, name: str, kinds: str) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        raise TypeError(
            f"{name} must be a torch.Tensor or numpy.ndarray, got {type(value).__name__}"
        )
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D [H, W], got {array.ndim} dim(s)")
    if array.dtype.kind not in kinds:
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    return array


def _as_seg_mask(seg_mask) -> np.ndarray:
    array = _as_2d_array(seg_mask, "segmentation mask", "iub")
    if array.dtype.kind != "i" and array.dtype.kind != "u" and array.dtype.kind != "b":
        raise ValueError(
            f"segmentation mask must contain integer class IDs, got dtype {array.dtype}"
        )
    if array.size and int(array.min()) < 0:
        raise ValueError("segmentation mask contains negative class IDs")
    return array.astype(np.int64)


def _as_depth(depth) -> np.ndarray:
    array = _as_2d_array(depth, "depth", "fiu")
    if array.dtype.kind == "b":
        raise ValueError("depth must not be boolean")
    return array.astype(np.float64)


def _prepare(seg_mask, depth) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    seg = _as_seg_mask(seg_mask)
    dep = _as_depth(depth)
    if seg.shape != dep.shape:
        raise ValueError(
            "segmentation and depth must have identical spatial dimensions, "
            f"got {seg.shape} vs {dep.shape}"
        )
    valid = np.isfinite(dep)
    analyzed = valid & (seg != VOID_ID)
    if not bool(valid.any()):
        raise ValueError("depth contains no finite values; nothing to analyze")
    if not bool(analyzed.any()):
        raise ValueError("segmentation mask contains no non-void pixels; nothing to analyze")
    return seg, dep, analyzed, int(analyzed.sum())


def _class_stats(seg: np.ndarray, dep: np.ndarray, analyzed: np.ndarray, analyzed_count: int, class_id: int) -> dict | None:
    mask = (seg == class_id) & analyzed
    count = int(mask.sum())
    if count == 0:
        return None
    values = dep[mask]
    return {
        "pixel_count": count,
        "pixel_ratio": float(count / analyzed_count),
        "mean_depth": float(values.mean()),
        "median_depth": float(np.median(values)),
        "min_depth": float(values.min()),
        "max_depth": float(values.max()),
    }


def validate_inputs(seg_mask, depth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a segmentation/depth pair and return their analyzed mask.

    Returns ``(seg_mask, depth, analyzed_mask)``. Raises a clear error on shape
    mismatch, non-integer segmentation, or an empty analyzed region.
    """
    seg, dep, analyzed, _ = _prepare(seg_mask, depth)
    return seg, dep, analyzed


def create_mask(seg_mask, class_id: int) -> np.ndarray:
    """Return the boolean pixel mask of a single semantic class."""
    if not isinstance(class_id, int) or isinstance(class_id, bool):
        raise TypeError(f"class_id must be an int, got {type(class_id).__name__}")
    seg = _as_seg_mask(seg_mask)
    if class_id < 0:
        raise ValueError(f"class_id must be non-negative, got {class_id}")
    return seg == class_id


def class_depth_stats(seg_mask, depth, class_id: int) -> dict | None:
    """Return depth statistics for one class, or ``None`` when the class is absent.

    Keys: ``pixel_count``, ``pixel_ratio`` (of analyzed pixels), ``mean_depth``,
    ``median_depth``, ``min_depth``, ``max_depth``.
    """
    seg, dep, analyzed, analyzed_count = _prepare(seg_mask, depth)
    return _class_stats(seg, dep, analyzed, analyzed_count, class_id)


def depth_regions(
    depth,
    quantiles: Sequence[float] | None = None,
    valid=None,
) -> tuple[np.ndarray, dict]:
    """Partition a relative-depth map into ``far`` / ``middle`` / ``near``.

    ``near`` = top quantiles[1] of the depth values (larger = closer under the
    inverse-depth convention), ``far`` = bottom quantiles[0]. Thresholds are
    derived from the prediction itself and are NOT meters.

    Returns ``(region_map, summary)`` where ``region_map`` uses
    ``REGION_FAR``/``REGION_MIDDLE``/``REGION_NEAR`` codes (``-1`` for invalid
    pixels) and ``summary`` contains per-region counts/ratios plus the
    ``{"low", "high"}`` boundaries.
    """
    q = DEFAULT_QUANTILES if quantiles is None else _validated_quantiles(quantiles)
    dep = _as_depth(depth)
    finite = np.isfinite(dep)
    mask = finite if valid is None else finite & np.asarray(valid, dtype=bool)
    if not bool(mask.any()):
        raise ValueError("no finite depth pixels to build regions")
    values = dep[mask]
    low, high = float(np.quantile(values, q[0])), float(np.quantile(values, q[1]))

    region_map = np.full(dep.shape, -1, dtype=np.int8)
    region_map[mask & (dep < low)] = REGION_FAR
    region_map[mask & (dep >= high)] = REGION_NEAR
    region_map[mask & ~(dep < low) & ~(dep >= high)] = REGION_MIDDLE

    total = int((region_map >= 0).sum())
    summary = {"thresholds": {"low": low, "high": high}}
    for name, code in zip(DEPTH_REGION_NAMES, (REGION_FAR, REGION_MIDDLE, REGION_NEAR)):
        count = int((region_map == code).sum())
        summary[name] = {"pixel_count": count, "pixel_ratio": float(count / total)}
    return region_map, summary


def _validated_quantiles(quantiles: Sequence[float]) -> tuple[float, float]:
    if len(quantiles) != 2:
        raise ValueError(f"exactly two region quantiles are required, got {len(quantiles)}")
    q0, q1 = float(quantiles[0]), float(quantiles[1])
    if not (0.0 < q0 < q1 < 1.0):
        raise ValueError(f"region quantiles must satisfy 0 < q0 < q1 < 1, got {(q0, q1)}")
    return q0, q1


def fuse(
    seg_mask,
    depth,
    classes: Sequence[int] | None = None,
    quantiles: Sequence[float] | None = None,
) -> FusionResult:
    """Combine a segmentation mask and relative depth into a :class:`FusionResult`.

    Args:
        seg_mask: ``[H, W]`` integer trainId mask.
        depth: ``[H, W]`` raw inverse relative depth.
        classes: Class IDs to summarize; default is all 19 trainIds. Missing
            classes are absent from ``per_class`` (never an error).
        quantiles: ``(q_low, q_high)`` region boundaries; default terciles.
    """
    seg, dep, analyzed, analyzed_count = _prepare(seg_mask, depth)

    requested = _CLASS_IDS_ALL if classes is None else tuple(int(c) for c in classes)
    per_class: dict[int, dict] = {}
    for class_id in requested:
        stats = _class_stats(seg, dep, analyzed, analyzed_count, class_id)
        if stats is not None:
            per_class[class_id] = stats

    region_map, summary = depth_regions(dep, quantiles=quantiles, valid=analyzed)
    return FusionResult(
        seg_mask=seg,
        depth=dep,
        valid_mask=np.isfinite(dep),
        analyzed_mask=analyzed,
        per_class=per_class,
        region_map=region_map,
        region_summary=summary,
        thresholds=(summary["thresholds"]["low"], summary["thresholds"]["high"]),
    )