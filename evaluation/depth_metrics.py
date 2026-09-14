"""Monocular depth evaluation metrics (MiDaS predictions vs KITTI ground truth).

KITTI ground truth is metric depth in meters (after the Step 03 preprocessing).
MiDaS produces *relative inverse depth*: its raw output has no metric scale, so
it must not be interpreted (or claimed) as meters.

Two evaluation representations are supported:

* **Relative** (``align=None``): the raw prediction is evaluated directly.
  The resulting error values are unit-less relative-depth errors, NOT meters.
* **Aligned metric** (``align="median"``): an explicit, evaluation-time
  median-scale alignment ``scale = median(gt) / median(pred)`` is applied
  before the metric is computed. This is a calibration step for evaluation
  only; MiDaS itself never predicts metric depth.

Only valid KITTI pixels are evaluated: finite GT with ``GT > 0`` and finite
prediction. Metrics that divide by prediction values (threshold accuracy,
median scaling) further require ``prediction > 0``. If no valid pixels exist a
clear ``ValueError`` is raised.

All accumulation is done in float64. NumPy arrays, (nested) Python sequences,
and PyTorch tensors are accepted.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import torch

_NUMERIC_KINDS = ("f", "i", "u")

_DELTA_THRESHOLDS: tuple[float, ...] = (1.25, 1.25**2, 1.25**3)


def _as_float_array(value: Any, name: str) -> np.ndarray:
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
    if array.ndim == 0 or array.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    return array.astype(np.float64)


def _as_bool_array(value: Any, name: str) -> np.ndarray:
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
    return array.astype(bool)


def _resolve(pred: Any, gt: Any, valid: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and return the flattened valid prediction/GT values.

    The mask keeps pixels with finite GT, ``GT > 0``, and finite prediction,
    AND-ed with the caller-provided ``valid`` mask when given. An all-empty
    mask raises ``ValueError``.
    """
    p = _as_float_array(pred, "prediction")
    g = _as_float_array(gt, "ground truth")
    if p.shape != g.shape:
        raise ValueError(
            f"prediction and ground truth must have the same spatial shape, "
            f"got {p.shape} vs {g.shape}"
        )
    if p.ndim != 2:
        raise ValueError(
            f"expected [H, W] depth maps, got {p.ndim} dim(s); evaluate one "
            f"image at a time"
        )

    mask = np.isfinite(g) & (g > 0.0) & np.isfinite(p)
    if valid is not None:
        v = _as_bool_array(valid, "valid mask")
        if v.shape != p.shape:
            raise ValueError(
                f"valid mask {v.shape} does not match the prediction shape {p.shape}"
            )
        mask &= v

    if not mask.any():
        raise ValueError(
            "no valid depth pixels: ground truth must be finite and positive "
            "and the prediction finite within the valid mask"
        )
    return p[mask], g[mask]


def rmse(pred: Any, gt: Any, valid: Any = None) -> float:
    """Root mean squared error over valid pixels: ``sqrt(mean((pred-gt)^2))``."""
    p, g = _resolve(pred, gt, valid)
    return float(np.sqrt(np.mean((p - g) ** 2.0)))


def mae(pred: Any, gt: Any, valid: Any = None) -> float:
    """Mean absolute error over valid pixels: ``mean(abs(pred-gt))``."""
    p, g = _resolve(pred, gt, valid)
    return float(np.mean(np.abs(p - g)))


def abs_relative_error(pred: Any, gt: Any, valid: Any = None) -> float:
    """Absolute relative error: ``mean(abs(pred-gt)/gt)`` (GT is positive)."""
    p, g = _resolve(pred, gt, valid)
    return float(np.mean(np.abs(p - g) / g))


def delta_accuracy(
    pred: Any,
    gt: Any,
    valid: Any = None,
    thresholds: Sequence[float] | None = _DELTA_THRESHOLDS,
) -> np.ndarray:
    """Threshold accuracy ``mean(max(pred/gt, gt/pred) < threshold)``.

    Returns one value per threshold in ``thresholds`` (default
    ``1.25^1, 1.25^2, 1.25^3``). Requires positive prediction and GT values,
    which are valid pixels by construction; any remaining non-positive
    prediction is excluded.
    """
    if thresholds is None:
        thresholds = _DELTA_THRESHOLDS
    if not isinstance(thresholds, (list, tuple, np.ndarray)):
        raise TypeError(f"thresholds must be a sequence, got {type(thresholds).__name__}")
    if len(thresholds) == 0:
        raise ValueError("at least one threshold is required")
    values = [float(t) for t in thresholds]
    if any(not np.isfinite(t) or t < 1.0 for t in values):
        raise ValueError(f"thresholds must be finite and >= 1.0, got {values}")

    p, g = _resolve(pred, gt, valid)
    keep = p > 0.0
    if not keep.any():
        raise ValueError(
            "no valid pixels with positive prediction for threshold accuracy"
        )
    p, g = p[keep], g[keep]
    ratio = np.maximum(p / g, g / p)
    return np.fromiter((float(np.mean(ratio < t)) for t in values), dtype=float, count=len(values))


def median_scale(
    pred: Any, gt: Any, valid: Any = None
) -> tuple[np.ndarray, float]:
    """Align a relative prediction to metric GT with median scaling.

    Returns ``(aligned_prediction, scale)`` where::

        scale = median(gt_valid) / median(pred_valid)
        aligned_prediction = prediction * scale

    Only valid pixels (finite, ``GT > 0``, finite/positive prediction) are used
    for the scale. The original prediction and GT are never modified; a new
    aligned copy is returned.
    """
    p, g = _resolve(pred, gt, valid)
    keep = p > 0.0
    if not keep.any():
        raise ValueError("no valid pixels with positive prediction for median scaling")
    p, g = p[keep], g[keep]

    scale = float(np.median(g) / np.median(p))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"median scaling produced a non-finite/non-positive scale: {scale}")

    base = _as_float_array(pred, "prediction")
    aligned = base * scale
    return aligned, scale


def evaluate_depth(
    pred: Any,
    gt: Any,
    valid: Any = None,
    align: str | None = "median",
) -> dict:
    """Compute all supported depth metrics for a prediction/GT pair.

    Args:
        pred: Predicted depth, ``[H, W]``.
        gt: KITTI metric ground-truth depth, ``[H, W]`` meters.
        valid: Optional ``[H, W]`` boolean mask AND-ed with the validity rules.
        align: ``"median"`` applies the evaluation-time median scale alignment
            before the metric; ``None`` evaluates the raw (relative) values.

    Returns:
        Dictionary with ``rmse``, ``mae``, ``abs_rel``, ``delta1``, ``delta2``,
        ``delta3`` and (when aligned) the computed ``scale``.

    For ``align=None`` the errors are relative-depth errors, not meters; only
    aligned metric evaluation is comparable to KITTI GT in meters.
    """
    if align == "median":
        aligned, scale = median_scale(pred, gt, valid)
        computed = aligned
    elif align is None:
        computed = _as_float_array(pred, "prediction")
        gt_arr = _as_float_array(gt, "ground truth")
        if computed.shape != gt_arr.shape:
            raise ValueError(
                f"prediction and ground truth must have the same spatial shape, "
                f"got {computed.shape} vs {gt_arr.shape}"
            )
        scale = None
    else:
        raise ValueError(f"unknown align mode {align!r}; use 'median' or None")

    deltas = delta_accuracy(computed, gt, valid)
    result = {
        "rmse": rmse(computed, gt, valid),
        "mae": mae(computed, gt, valid),
        "abs_rel": abs_relative_error(computed, gt, valid),
        "delta1": float(deltas[0]),
        "delta2": float(deltas[1]),
        "delta3": float(deltas[2]),
    }
    if scale is not None:
        result["scale"] = scale
    return result