import numpy as np
import pytest
import torch

from evaluation.segmentation_metrics import (
    class_accuracy,
    class_dice,
    class_iou,
    confusion_matrix,
    dice_score,
    evaluate_segmentation,
    intersection_over_union,
    mean_dice,
    mean_iou,
    pixel_accuracy,
)


def _t(tensor):
    return torch.as_tensor(tensor)


def test_perfect_prediction():
    gt = [[0, 1], [2, 3]]
    pred = [[0, 1], [2, 3]]
    metrics = evaluate_segmentation(pred, gt, num_classes=4)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["mean_dice"] == pytest.approx(1.0)
    assert np.allclose(metrics["iou_per_class"], 1.0)
    assert np.allclose(metrics["dice_per_class"], 1.0)


def test_completely_wrong_prediction():
    gt = np.asarray([[0, 0], [0, 0]])
    pred = np.asarray([[1, 1], [1, 1]])
    metrics = evaluate_segmentation(pred, gt, num_classes=2)
    assert metrics["pixel_accuracy"] == pytest.approx(0.0)
    assert metrics["miou"] == pytest.approx(0.0)
    assert metrics["mean_dice"] == pytest.approx(0.0)
    assert np.allclose(metrics["iou_per_class"], 0.0)


def test_ignore_index_is_excluded():
    gt = [[0, 1], [2, 255]]
    pred = [[0, 1], [0, 2]]
    metrics = evaluate_segmentation(pred, gt, num_classes=3)
    assert metrics["pixel_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["iou_per_class"][0] == pytest.approx(0.5)  # TP=1, FP=1, FN=0
    assert metrics["iou_per_class"][1] == pytest.approx(1.0)
    assert metrics["iou_per_class"][2] == pytest.approx(0.0)  # FN=1, no TP/FP
    assert metrics["miou"] == pytest.approx(0.5)
    assert np.allclose(metrics["dice_per_class"], [2 / 3, 1.0, 0.0])


def test_ignore_index_exact_example():
    gt = [[0, 0, 1, 255]]
    pred = [[0, 1, 1, 2]]
    assert pixel_accuracy(pred, gt) == pytest.approx(2.0 / 3.0)
    assert confusion_matrix(pred, gt, num_classes=3).shape == (3, 3)
    assert pixel_accuracy(pred, gt, ignore_index=None) == pytest.approx(2.0 / 4.0)


def test_pixel_accuracy_manual():
    gt = np.asarray([[0, 0], [1, 1]])
    pred = np.asarray([[0, 1], [1, 0]])
    assert pixel_accuracy(pred, gt) == pytest.approx(0.5)


def test_iou_manual():
    gt = np.asarray([[0, 0], [1, 1]])
    pred = np.asarray([[0, 1], [1, 0]])
    ious = class_iou(pred, gt, num_classes=2)
    assert ious[0] == pytest.approx(1.0 / 3.0)
    assert ious[1] == pytest.approx(1.0 / 3.0)
    assert mean_iou(pred, gt, num_classes=2) == pytest.approx(1.0 / 3.0)
    assert np.allclose(intersection_over_union(pred, gt, num_classes=2), ious)


def test_dice_manual():
    gt = np.asarray([[0, 0], [1, 1]])
    pred = np.asarray([[0, 1], [1, 0]])
    dices = class_dice(pred, gt, num_classes=2)
    assert dices[0] == pytest.approx(0.5)
    assert dices[1] == pytest.approx(0.5)
    assert mean_dice(pred, gt, num_classes=2) == pytest.approx(0.5)
    assert np.allclose(dice_score(pred, gt, num_classes=2), dices)


def test_multiple_classes():
    gt = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    pred = np.asarray([[0, 1, 1], [2, 0, 0]], dtype=np.int64)
    metrics = evaluate_segmentation(pred, gt, num_classes=3)
    tp = np.asarray([2, 1, 1], dtype=np.float64)
    actual = np.asarray([2, 2, 2], dtype=np.float64)
    predicted = np.asarray([3, 2, 1], dtype=np.float64)
    expected_iou = tp / (actual + predicted - tp)
    assert np.allclose(metrics["iou_per_class"], expected_iou)
    assert metrics["miou"] == pytest.approx(expected_iou.mean())
    assert metrics["mean_dice"] == pytest.approx((2 * tp / (actual + predicted)).mean())


def test_missing_class_is_not_padded_to_one():
    gt = np.asarray([[0, 0], [1, 1]])
    pred = np.asarray([[0, 0], [1, 1]])
    metrics = evaluate_segmentation(pred, gt, num_classes=3)
    assert metrics["iou_per_class"][2] == pytest.approx(0.0)
    assert metrics["dice_per_class"][2] == pytest.approx(0.0)
    assert metrics["miou"] == pytest.approx(1.0)      # defined classes only
    assert metrics["mean_dice"] == pytest.approx(1.0)
    assert metrics["miou"] != pytest.approx(2.0 / 3.0)  # not divided by 3


def test_all_undefined_returns_zero():
    gt = np.full((2, 2), 255, dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)
    metrics = evaluate_segmentation(pred, gt, num_classes=3)
    assert metrics["pixel_accuracy"] == pytest.approx(0.0)
    assert metrics["miou"] == pytest.approx(0.0)
    assert metrics["mean_dice"] == pytest.approx(0.0)
    assert np.allclose(metrics["iou_per_class"], 0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same spatial shape"):
        pixel_accuracy(np.zeros((2, 3), dtype=np.int64), np.zeros((3, 2), dtype=np.int64))


def test_numpy_and_torch_agree():
    gt_np = np.random.randint(0, 4, size=(8, 8)).astype(np.int64)
    pred_np = np.random.randint(0, 4, size=(8, 8)).astype(np.int64)
    np_metrics = evaluate_segmentation(pred_np, gt_np, num_classes=4)
    torch_metrics = evaluate_segmentation(_t(pred_np), _t(gt_np), num_classes=4)
    assert np_metrics["pixel_accuracy"] == pytest.approx(torch_metrics["pixel_accuracy"])
    assert np_metrics["miou"] == pytest.approx(torch_metrics["miou"])
    assert np_metrics["mean_dice"] == pytest.approx(torch_metrics["mean_dice"])
    assert np.allclose(np_metrics["iou_per_class"], torch_metrics["iou_per_class"])


def test_torch_input_works():
    gt = torch.tensor([[0, 1, 255], [2, 2, 1]], dtype=torch.int64)
    pred = torch.tensor([[0, 1, 2], [2, 2, 1]], dtype=torch.int64)
    assert pixel_accuracy(pred, gt) == pytest.approx(1.0)
    assert mean_iou(pred, gt, num_classes=3) == pytest.approx(1.0)


def test_batch_input_works():
    gt1 = _t([[0, 0], [1, 1]])
    pred1 = _t([[0, 1], [1, 0]])
    gt2 = _t([[0, 1], [2, 2]])
    pred2 = _t([[0, 1], [2, 2]])
    batch_metrics = evaluate_segmentation(
        torch.stack([pred1, pred2]), torch.stack([gt1, gt2]), num_classes=3
    )
    merged_metrics = evaluate_segmentation(
        torch.cat([pred1, pred2], dim=1), torch.cat([gt1, gt2], dim=1), num_classes=3
    )
    for key in ("pixel_accuracy", "miou", "mean_dice"):
        assert batch_metrics[key] == pytest.approx(merged_metrics[key])
    assert batch_metrics["pixel_accuracy"] == pytest.approx(0.75)  # (2 + 4) / 8


def test_metrics_stay_within_unit_range():
    rng = np.random.default_rng(0)
    for _ in range(20):
        gt = rng.integers(0, 5, size=(16, 16))
        pred = rng.integers(0, 5, size=(16, 16))
        gt = np.where(gt == 4, 255, gt)  # introduce some ignore pixels
        pred = np.where(pred == 4, 3, pred)
        metrics = evaluate_segmentation(pred, gt, num_classes=4)
        assert 0.0 <= metrics["pixel_accuracy"] <= 1.0
        assert 0.0 <= metrics["miou"] <= 1.0
        assert 0.0 <= metrics["mean_dice"] <= 1.0
        assert np.all((metrics["iou_per_class"] >= 0.0) & (metrics["iou_per_class"] <= 1.0))
        assert np.all((metrics["dice_per_class"] >= 0.0) & (metrics["dice_per_class"] <= 1.0))
        assert np.all((metrics["class_accuracy"] >= 0.0) & (metrics["class_accuracy"] <= 1.0))


def test_invalid_num_classes_raises():
    gt = np.zeros((2, 2), dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="num_classes"):
        mean_iou(pred, gt, num_classes=0)
    with pytest.raises(ValueError, match="num_classes"):
        mean_iou(pred, gt, num_classes="19")


def test_invalid_ignore_index_raises():
    gt = np.zeros((2, 2), dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="ignore_index"):
        pixel_accuracy(pred, gt, ignore_index=-1)
    with pytest.raises(ValueError, match="ignore_index"):
        pixel_accuracy(pred, gt, ignore_index=1.5)


def test_out_of_range_predictions_raise():
    gt = np.zeros((2, 2), dtype=np.int64)
    pred = np.full((2, 2), 3, dtype=np.int64)
    with pytest.raises(ValueError, match="outside"):
        mean_iou(pred, gt, num_classes=3)


def test_out_of_range_targets_raise():
    gt = np.asarray([[4, 0], [0, 0]], dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="outside"):
        mean_iou(pred, gt, num_classes=3)


def test_non_integer_input_raises():
    gt = np.zeros((2, 2), dtype=np.float32)
    pred = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="integer class IDs"):
        pixel_accuracy(pred, gt)


def test_unsupported_type_raises():
    with pytest.raises(TypeError, match="torch.Tensor or numpy.ndarray"):
        pixel_accuracy("not-an-array", "not-an-array")


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError, match="dims"):
        pixel_accuracy(np.zeros(5, dtype=np.int64), np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError, match="dims"):
        pixel_accuracy(np.zeros((1, 2, 3, 4), dtype=np.int64), np.zeros((1, 2, 3, 4), dtype=np.int64))


def test_class_accuracy():
    gt = np.asarray([[0, 0], [1, 1]])
    pred = np.asarray([[0, 1], [1, 0]])
    acc = class_accuracy(pred, gt, num_classes=2)
    assert acc[0] == pytest.approx(0.5)  # TP=1 / actual=2
    assert acc[1] == pytest.approx(0.5)


def test_manual_verification_example():
    gt = np.asarray([[0, 0], [1, 1]])
    pred = np.asarray([[0, 1], [1, 0]])
    metrics = evaluate_segmentation(pred, gt, num_classes=2)
    assert metrics["pixel_accuracy"] == pytest.approx(2 / 4)
    assert np.allclose(metrics["iou_per_class"], [1 / 3, 1 / 3])
    assert metrics["miou"] == pytest.approx(1 / 3)
    assert np.allclose(metrics["dice_per_class"], [1 / 2, 1 / 2])
    assert metrics["mean_dice"] == pytest.approx(1 / 2)