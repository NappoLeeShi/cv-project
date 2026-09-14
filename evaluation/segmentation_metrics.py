"""Semantic segmentation evaluation metrics.

Metrics are computed from a single reusable confusion matrix (only
ground-truth-valid pixels; ``ignore_index`` never becomes a class). Both
PyTorch tensors and NumPy arrays are accepted and converted to a common
integer representation, so no metric logic is duplicated.

Convention for undefined classes (``TP + FP + FN == 0``): per-class IoU/Dice
are ``0.0`` and the class is excluded from the means, so an absent class can
never silently inflate mIoU / mean Dice. If every class is undefined the mean
is ``0.0``.
"""

from __future__ import annotations

import numpy as np
import torch

_UINT = {"u", "i"}


def _as_numpy(value, name: str) -> np.ndarray:
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
    if array.dtype.kind not in _UINT:
        raise ValueError(f"{name} must contain integer class IDs, got dtype {array.dtype}")
    return array


def _validate(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int | None,
    ignore_index: int | None,
) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same spatial shape, "
            f"got {pred.shape} vs {target.shape}"
        )
    if pred.ndim not in (2, 3):
        raise ValueError(
            f"expected [H, W] or [B, H, W] segmentation maps, got {pred.ndim} dims"
        )
    if num_classes is not None and (
        not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1
    ):
        raise ValueError(f"num_classes must be a positive int, got {num_classes!r}")
    if ignore_index is not None and (
        not isinstance(ignore_index, int) or isinstance(ignore_index, bool) or ignore_index < 0
    ):
        raise ValueError(f"ignore_index must be a non-negative int or None, got {ignore_index!r}")

    if num_classes is None:
        return

    valid = target != ignore_index if ignore_index is not None else np.ones_like(target, dtype=bool)
    if np.any(pred < 0) or np.any(pred >= num_classes):
        raise ValueError(
            f"prediction contains class IDs outside [0, {num_classes}); "
            f"range was [{pred.min()}, {pred.max()}]"
        )
    target_valid = target[valid]
    if np.any(target_valid < 0) or np.any(target_valid >= num_classes):
        raise ValueError(
            f"target contains class IDs outside [0, {num_classes}) (excluding "
            f"ignore_index); range was [{target_valid.min()}, {target_valid.max()}]"
        )


def confusion_matrix(
    pred,
    target,
    num_classes: int = 19,
    ignore_index: int | None = 255,
) -> np.ndarray:
    """Return the ``(num_classes, num_classes)`` confusion matrix.

    Rows are ground-truth classes, columns predicted classes. Pixels whose
    ground truth equals ``ignore_index`` are excluded entirely; ``None``
    disables the ignore masking.
    """
    pred_arr = _as_numpy(pred, "prediction")
    target_arr = _as_numpy(target, "target")
    _validate(pred_arr, target_arr, num_classes, ignore_index)

    if ignore_index is not None:
        valid = target_arr != ignore_index
        pred_arr = pred_arr[valid]
        target_arr = target_arr[valid]

    flat = target_arr.astype(np.int64) * num_classes + pred_arr.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _per_class_arrays(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    return actual, predicted, tp


def _iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    actual, predicted, tp = _per_class_arrays(cm)
    denom = actual + predicted - tp
    return np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)


def _dice_from_confusion(cm: np.ndarray) -> np.ndarray:
    actual, predicted, tp = _per_class_arrays(cm)
    denom = actual + predicted
    return np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def _accuracy_from_confusion(cm: np.ndarray) -> np.ndarray:
    actual, _, tp = _per_class_arrays(cm)
    return np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)


def _mean_valid(values: np.ndarray, denom_mask: np.ndarray, cm: np.ndarray) -> float:
    valid = denom_mask & (cm.sum() > 0)
    chosen = values[valid]
    if chosen.size == 0:
        return 0.0
    return float(chosen.mean())


def pixel_accuracy(pred, target, ignore_index: int | None = 255) -> float:
    """Fraction of correctly classified valid pixels (``0.0`` if none valid)."""
    pred_arr = _as_numpy(pred, "prediction")
    target_arr = _as_numpy(target, "target")
    _validate(pred_arr, target_arr, num_classes=None, ignore_index=ignore_index)

    valid = target_arr != ignore_index if ignore_index is not None else np.ones_like(target_arr, dtype=bool)
    total = int(valid.sum())
    if total == 0:
        return 0.0
    correct = int((pred_arr[valid] == target_arr[valid]).sum())
    return float(correct / total)


def class_iou(pred, target, num_classes: int = 19, ignore_index: int | None = 255) -> np.ndarray:
    """Per-class IoU; undefined classes are ``0.0``."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    return _iou_from_confusion(cm)


def mean_iou(pred, target, num_classes: int = 19, ignore_index: int | None = 255) -> float:
    """Mean IoU over classes with at least one pixel (``0.0`` if none)."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    ious = _iou_from_confusion(cm)
    actual, predicted, _ = _per_class_arrays(cm)
    return _mean_valid(ious, (actual + predicted - np.diag(cm).astype(np.float64)) > 0, cm)


def class_dice(pred, target, num_classes: int = 19, ignore_index: int | None = 255) -> np.ndarray:
    """Per-class Dice; undefined classes are ``0.0``."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    return _dice_from_confusion(cm)


def mean_dice(pred, target, num_classes: int = 19, ignore_index: int | None = 255) -> float:
    """Mean Dice over classes with at least one pixel (``0.0`` if none)."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    dices = _dice_from_confusion(cm)
    actual, predicted, _ = _per_class_arrays(cm)
    return _mean_valid(dices, (actual + predicted) > 0, cm)


def class_accuracy(pred, target, num_classes: int = 19, ignore_index: int | None = 255) -> np.ndarray:
    """Per-class accuracy ``TP / (TP + FN)``; absent classes are ``0.0``."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    return _accuracy_from_confusion(cm)


intersection_over_union = class_iou
dice_score = class_dice


def evaluate_segmentation(
    pred,
    target,
    num_classes: int = 19,
    ignore_index: int | None = 255,
) -> dict:
    """Return all segmentation metrics for a prediction/target pair.

    Keys: ``pixel_accuracy``, ``iou_per_class``, ``miou``, ``dice_per_class``,
    ``mean_dice``, ``class_accuracy``.
    """
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    ious = _iou_from_confusion(cm)
    dices = _dice_from_confusion(cm)
    actual, predicted, tp = _per_class_arrays(cm)
    iou_denom = actual + predicted - tp
    dice_denom = actual + predicted
    return {
        "pixel_accuracy": pixel_accuracy(pred, target, ignore_index),
        "iou_per_class": ious,
        "miou": _mean_valid(ious, iou_denom > 0, cm),
        "dice_per_class": dices,
        "mean_dice": _mean_valid(dices, dice_denom > 0, cm),
        "class_accuracy": _accuracy_from_confusion(cm),
    }