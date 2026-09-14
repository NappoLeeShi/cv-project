"""High-level scene understanding built on top of :mod:`.fusion`.

Consumes a :class:`~scene_understanding.fusion.FusionResult` and produces a
JSON-serializable, interpretable traffic-scene report. Depth statements are made
exclusively in relative terms (near / middle / far regions), never in meters,
because the underlying MiDaS depth is relative inverse depth.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from scene_understanding.fusion import (
    CITYSCAPES_TRAINID_NAMES,
    DEFAULT_DRIVABLE_CLASSES,
    DEFAULT_ROI_CLASSES,
    FusionResult,
    fuse,
)
from utils.config import get

_PERSON_ID = 11


def _name(class_id: int) -> str:
    return CITYSCAPES_TRAINID_NAMES.get(int(class_id), str(int(class_id)))


def _region_label(median_depth: float, low: float, high: float) -> str:
    if median_depth >= high:
        return "near"
    if median_depth >= low:
        return "middle"
    return "far"


def _pick(
    cfg: dict | None, key: str, default: Any, override: Any
) -> Any:
    if override is not None:
        return override
    if cfg is not None:
        value = get(cfg, key)
        if value is not None:
            return tuple(value) if isinstance(value, list) else value
    return default


def analyze_fusion(
    fr: FusionResult,
    cfg: dict | None = None,
    roi_classes: Sequence[int] | None = None,
    drivable_classes: Sequence[int] | None = None,
) -> dict:
    """Translate a :class:`FusionResult` into a structured scene report.

    Args:
        fr: Output of :func:`scene_understanding.fusion.fuse`.
        roi_classes: Dynamic classes (default person..bicycle, matching the
            ``fusion.roi_classes`` config).
        drivable_classes: Ground classes (default road + sidewalk).

    Returns a plain dictionary suitable for ``json.dumps``.
    """
    roi = tuple(int(c) for c in _pick(cfg, "fusion.roi_classes", DEFAULT_ROI_CLASSES, roi_classes))
    drivable = tuple(
        int(c) for c in _pick(cfg, "fusion.drivable_classes", DEFAULT_DRIVABLE_CLASSES, drivable_classes)
    )
    low, high = fr.thresholds

    height, width = fr.seg_mask.shape
    analyzed = int(fr.analyzed_mask.sum())
    per_class = fr.per_class
    present_ids = sorted(per_class)

    scene = {
        "height": height,
        "width": width,
        "analyzed_pixels": analyzed,
        "num_semantic_classes": len(present_ids),
        "depth_convention": "inverse_relative_larger_closer",
        "depth_thresholds": {"low": low, "high": high},
    }

    semantic_distribution = {
        _name(class_id): per_class[class_id]["pixel_count"] for class_id in present_ids
    }
    object_count = sum(per_class[c]["pixel_count"] for c in present_ids if c in roi)
    scene["semantic_distribution"] = semantic_distribution
    scene["object_class_pixel_ratio"] = float(object_count / analyzed) if analyzed else 0.0

    depth_distribution = {
        name: fr.region_summary[name]["pixel_ratio"]
        for name in ("near", "middle", "far")
    }

    regions = {
        _name(class_id): {k: v for k, v in per_class[class_id].items()}
        for class_id in present_ids
    }

    traffic = _traffic_context(per_class, roi, drivable, analyzed, low, high)

    return {
        "scene": scene,
        "semantic_distribution": semantic_distribution,
        "depth_distribution": depth_distribution,
        "regions": regions,
        "traffic_context": traffic,
        "interpretation": _interpretation(traffic),
    }


def _traffic_context(
    per_class: dict[int, dict],
    roi: tuple[int, ...],
    drivable: tuple[int, ...],
    analyzed: int,
    low: float,
    high: float,
) -> dict:
    label = lambda median: _region_label(median, low, high)  # noqa: E731

    vehicles = {}
    for class_id in roi:
        if class_id == _PERSON_ID:
            continue
        if class_id in per_class:
            stats = per_class[class_id]
            vehicles[_name(class_id)] = {
                "pixel_count": stats["pixel_count"],
                "median_depth": stats["median_depth"],
                "proximity": label(stats["median_depth"]),
            }

    pedestrians = {}
    if _PERSON_ID in per_class:
        stats = per_class[_PERSON_ID]
        pedestrians["person"] = {
            "pixel_count": stats["pixel_count"],
            "median_depth": stats["median_depth"],
            "proximity": label(stats["median_depth"]),
        }

    road = {}
    for class_id in drivable:
        if class_id in per_class:
            stats = per_class[class_id]
            road[_name(class_id)] = {
                "pixel_count": stats["pixel_count"],
                "median_depth": stats["median_depth"],
                "proximity": label(stats["median_depth"]),
            }

    drivable_count = sum(per_class[c]["pixel_count"] for c in drivable if c in per_class)
    drivable_median = _median_over_classes(per_class, [c for c in drivable if c in per_class])
    traffic: dict = {
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "road": road,
        "drivable_coverage_ratio": float(drivable_count / analyzed) if analyzed else 0.0,
        "drivable_median_depth": drivable_median,
    }

    present_dynamic = [c for c in roi if c in per_class]
    traffic["dynamic_object_count"] = len(present_dynamic)
    if present_dynamic:
        closest = max(present_dynamic, key=lambda c: per_class[c]["median_depth"])
        closest_stats = per_class[closest]
        traffic["nearest_dynamic_class"] = {
            "class": _name(closest),
            "median_depth": closest_stats["median_depth"],
            "proximity": label(closest_stats["median_depth"]),
        }
    else:
        traffic["nearest_dynamic_class"] = None

    for class_id in present_dynamic:
        if class_id in drivable:
            pass  # drivable classes are handled by the road block above

    return traffic


def _median_over_classes(per_class: dict[int, dict], class_ids: list[int]) -> float | None:
    values = [per_class[c]["median_depth"] for c in class_ids]
    if not values:
        return None
    return float(np.median(values))


def _interpretation(traffic: dict) -> list[str]:
    lines: list[str] = []
    nearest = traffic.get("nearest_dynamic_class")
    if nearest and nearest["class"] is not None:
        lines.append(
            f"The nearest dynamic class is '{nearest['class']}' "
            f"({nearest['proximity']} relative-depth region)."
        )
    for name, stats in traffic.get("vehicles", {}).items():
        lines.append(
            f"Vehicle class '{name}' appears in the {stats['proximity']} "
            "relative-depth region."
        )
    person = traffic.get("pedestrians", {}).get("person")
    if person:
        lines.append(
            f"Person pixels appear in the {person['proximity']} relative-depth region."
        )
    road = traffic.get("road", {})
    if road:
        names = ", ".join(road)
        lines.append(f"Drivable areas ({names}) cover the lower/middle scene relative-depth region.")
    return lines


def analyze_maps(
    seg_mask,
    depth,
    cfg: dict | None = None,
    roi_classes: Sequence[int] | None = None,
    quantiles: Sequence[float] | None = None,
) -> dict:
    """Analyze an already-produced segmentation mask + relative depth map.

    Convenience wrapper around :func:`scene_understanding.fusion.fuse` and
    :func:`analyze_fusion`. Region quantiles fall back to
    ``fusion.region_quantiles`` from ``cfg`` when not given explicitly.
    """
    if quantiles is None and cfg is not None:
        configured = get(cfg, "fusion.region_quantiles")
        if configured is not None:
            quantiles = tuple(configured)
    fr = fuse(seg_mask, depth, quantiles=quantiles)
    return analyze_fusion(fr, cfg=cfg, roi_classes=roi_classes)


def analyze_scene(
    image,
    seg_predictor,
    depth_predictor,
    cfg: dict | None = None,
    roi_classes: Sequence[int] | None = None,
    quantiles: Sequence[float] | None = None,
) -> dict:
    """Run the whole inference-only path on a single street image.

    ``image`` is passed unchanged to ``seg_predictor.predict`` and
    ``depth_predictor.predict`` (every object exposing a ``.predict`` callable
    is accepted, so U-Net and MiDaS wrappers plug in directly). The same exact
    image feeds both models; predictions are fused and analyzed.

    Ground truth is never required.
    """
    seg = seg_predictor.predict(image)
    depth = depth_predictor.predict(image)
    return analyze_maps(seg, depth, cfg=cfg, roi_classes=roi_classes, quantiles=quantiles)