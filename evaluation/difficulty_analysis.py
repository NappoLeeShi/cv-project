"""Rule-based scene complexity (difficulty) scoring.

Computes a transparent ``difficulty_score`` from fusion and scene-analysis
indicators.  The score is NOT a ground-truth difficulty label; it is a
``pipeline difficulty score`` / ``scene complexity score`` derived from
weighted, normalized indicators.

Default indicators (5):

1. **segmentation uncertainty** — ``1 - mean_confidence`` from the U-Net
   softmax.  Higher uncertainty → harder scene.
2. **depth variation** — coefficient of variation (std / mean) of inverse
   depth.  More variation → more depth discontinuities → harder.
3. **scene complexity** — number of distinct semantic classes present, divided
   by the total possible (19 Cityscapes trainIds).  More classes → harder.
4. **foreground fraction** — fraction of pixels in the near depth region.
   More foreground clutter → harder.
5. **object density** — ratio of ROI (dynamic-object) class pixels to all
   analyzed pixels.  More objects → harder.

Each indicator is clamped to [0, 1] and combined with configurable weights:

    difficulty_score = Σ weight_i × indicator_i

Thresholds (configurable via ``difficulty.bins``):

    score ≤ bins.easy   → "easy"
    score ≤ bins.medium → "medium"
    otherwise           → "hard"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scene_understanding.fusion import FusionResult, CITYSCAPES_TRAINID_NAMES

DEFAULT_WEIGHTS: list[float] = [0.25, 0.20, 0.20, 0.20, 0.15]
DEFAULT_BINS: dict[str, float] = {"easy": 0.4, "medium": 0.7}
_NUM_CLASSES_TOTAL = 19


@dataclass
class DifficultyResult:
    """Structured difficulty analysis output.

    Attributes:
        score: Weighted difficulty score in [0, 1].
        level: One of ``"easy"``, ``"medium"``, ``"hard"``.
        components: Per-indicator raw and normalized values.
    """

    score: float
    level: str
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "level": self.level,
            "components": dict(self.components),
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _segmentation_confidence_indicator(
    mean_confidence: float | None,
) -> float:
    """1 - mean_confidence.  None (unavailable) → 0.5 (neutral)."""
    if mean_confidence is None:
        return 0.5
    return _clamp(1.0 - float(mean_confidence))


def _depth_variation_indicator(depth: np.ndarray) -> float:
    """Coefficient of variation of finite inverse depth, clamped to [0, 1]."""
    finite = np.isfinite(depth)
    values = depth[finite]
    if values.size == 0:
        return 0.0
    mean_val = float(values.mean())
    if mean_val == 0.0:
        return 0.0
    cv = float(values.std()) / abs(mean_val)
    return _clamp(cv)


def _scene_complexity_indicator(num_classes_present: int) -> float:
    """Fraction of the 19 Cityscapes trainIds that are present."""
    return _clamp(num_classes_present / _NUM_CLASSES_TOTAL)


def _foreground_fraction_indicator(region_summary: dict) -> float:
    """Fraction of pixels in the 'near' depth region."""
    near = region_summary.get("near", {})
    return _clamp(near.get("pixel_ratio", 0.0))


def _object_density_indicator(
    fusion_result: FusionResult,
    roi_classes: tuple[int, ...] | None = None,
) -> float:
    """Fraction of analyzed pixels belonging to ROI (dynamic) classes."""
    analyzed_count = int(fusion_result.analyzed_mask.sum())
    if analyzed_count == 0:
        return 0.0
    if roi_classes is None:
        roi_classes = tuple(range(11, 19))
    count = sum(
        fusion_result.per_class[c]["pixel_count"]
        for c in roi_classes
        if c in fusion_result.per_class
    )
    return _clamp(count / analyzed_count)


_INDICATOR_NAMES = (
    "segmentation_uncertainty",
    "depth_variation",
    "scene_complexity",
    "foreground_fraction",
    "object_density",
)


def compute_difficulty(
    fusion_result: FusionResult,
    *,
    scene_report: dict | None = None,
    mean_confidence: float | None = None,
    roi_classes: tuple[int, ...] | None = None,
    weights: list[float] | None = None,
    bins: dict[str, float] | None = None,
) -> DifficultyResult:
    """Compute a rule-based difficulty score for one image.

    Args:
        fusion_result: Output of :func:`scene_understanding.fusion.fuse`.
        scene_report: Optional scene analysis report (used to extract
            ``num_semantic_classes`` when available).
        mean_confidence: Optional per-pixel classification confidence
            (max softmax probability) averaged over the image.
        roi_classes: Dynamic-object class IDs (default person..bicycle).
        weights: Five indicator weights (must sum to 1.0).
        bins: ``{"easy": threshold, "medium": threshold}``.

    Returns:
        A :class:`DifficultyResult` with score, level, and component details.
    """
    if weights is None:
        weights = list(DEFAULT_WEIGHTS)
    if bins is None:
        bins = dict(DEFAULT_BINS)
    if len(weights) != 5:
        raise ValueError(f"exactly 5 weights required, got {len(weights)}")

    # --- raw indicators ---
    ind_seg = _segmentation_confidence_indicator(mean_confidence)
    ind_depth = _depth_variation_indicator(fusion_result.depth)
    num_classes = len(fusion_result.per_class)
    if scene_report is not None:
        num_classes = scene_report.get("scene", {}).get(
            "num_semantic_classes", num_classes
        )
    ind_complexity = _scene_complexity_indicator(num_classes)
    ind_fg = _foreground_fraction_indicator(fusion_result.region_summary)
    ind_obj = _object_density_indicator(fusion_result, roi_classes)

    raw = {
        "segmentation_uncertainty": ind_seg,
        "depth_variation": ind_depth,
        "scene_complexity": ind_complexity,
        "foreground_fraction": ind_fg,
        "object_density": ind_obj,
    }

    # --- weighted score ---
    score = sum(w * v for w, v in zip(weights, raw.values()))
    score = _clamp(float(score))

    # --- classify ---
    easy_thresh = float(bins.get("easy", DEFAULT_BINS["easy"]))
    medium_thresh = float(bins.get("medium", DEFAULT_BINS["medium"]))
    if score <= easy_thresh:
        level = "easy"
    elif score <= medium_thresh:
        level = "medium"
    else:
        level = "hard"

    return DifficultyResult(
        score=round(score, 6),
        level=level,
        components=raw,
    )


def difficulty_from_report(
    fusion_result: FusionResult,
    scene_report: dict,
    *,
    mean_confidence: float | None = None,
    roi_classes: tuple[int, ...] | None = None,
    weights: list[float] | None = None,
    bins: dict[str, float] | None = None,
) -> dict:
    """Convenience wrapper returning a JSON-serializable dict."""
    result = compute_difficulty(
        fusion_result,
        scene_report=scene_report,
        mean_confidence=mean_confidence,
        roi_classes=roi_classes,
        weights=weights,
        bins=bins,
    )
    return result.to_dict()
