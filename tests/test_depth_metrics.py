import numpy as np
import pytest
import torch

from evaluation.depth_metrics import (
    abs_relative_error,
    delta_accuracy,
    evaluate_depth,
    mae,
    median_scale,
    rmse,
)

GT = np.array([[3.0, 3.0], [3.0, 3.0]])
PRED_ZERO = np.zeros((2, 2), dtype=np.float64)

# gt = 10 everywhere; ratios max(pred/gt, gt/pred) = [2.0, 1.2, 1.0, 1.7]
GT4 = np.array([[10.0, 10.0], [10.0, 10.0]])
PRED4 = np.array([[5.0, 12.0], [10.0, 17.0]])


# 1. RMSE known example
def test_rmse_known_example():
    # diffs are all 3 -> rmse 3
    assert rmse(PRED_ZERO, GT) == pytest.approx(3.0)


# 2. MAE known example
def test_mae_known_example():
    assert mae(PRED_ZERO, GT) == pytest.approx(3.0)


# 3. AbsRel known example
def test_abs_relative_error_known_example():
    assert abs_relative_error(PRED_ZERO, GT) == pytest.approx(1.0)


# 4/5/6. delta1 / delta2 / delta3
def test_delta1_known_example():
    deltas = delta_accuracy(PRED4, GT4)
    assert deltas[0] == pytest.approx(0.5)


def test_delta2_known_example():
    deltas = delta_accuracy(PRED4, GT4)
    assert deltas[1] == pytest.approx(0.5)


def test_delta3_known_example():
    deltas = delta_accuracy(PRED4, GT4)
    assert deltas[2] == pytest.approx(0.75)


# 7. valid mask
def test_valid_mask_restricts_evaluation():
    gt = np.array([[3.0, 3.0], [3.0, 0.0]])
    pred = np.array([[0.0, 0.0], [0.0, 99.0]])
    assert rmse(pred, gt) == pytest.approx(3.0)


def test_explicit_valid_mask_anded_with_rules():
    gt = GT
    pred = PRED_ZERO
    valid = np.array([[True, False], [True, True]])
    # valid pixels: (0,0), (1,0), (1,1) -> error 3 each
    assert rmse(pred, gt, valid) == pytest.approx(3.0)


# 8. invalid GT ignored
def test_invalid_gt_ignored():
    gt = np.array([[3.0, 3.0], [3.0, np.inf]])
    pred = PRED_ZERO
    assert abs_relative_error(pred, gt) == pytest.approx(1.0)


def test_negative_gt_ignored():
    gt = np.array([[3.0, -1.0], [4.0, 0.0]])
    pred = PRED_ZERO
    # only gt=3 and gt=4 are valid; abs_rel = mean(3/3, 4/4) = 1
    assert abs_relative_error(pred, gt) == pytest.approx(1.0)


# 9. invalid prediction ignored
def test_invalid_prediction_ignored():
    pred = np.array([[0.0, 0.0], [0.0, np.nan]])
    assert abs_relative_error(pred, GT) == pytest.approx(1.0)
    pred_inf = np.array([[0.0, 0.0], [0.0, np.inf]])
    assert abs_relative_error(pred_inf, GT) == pytest.approx(1.0)


# 10. median scaling known example
def test_median_scale_known_example():
    pred = np.array([[10.0, 20.0], [30.0, 40.0]])
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    aligned, scale = median_scale(pred, gt)
    assert scale == pytest.approx(0.1)
    assert np.allclose(aligned, gt)


# 11. median scaling does not mutate input
def test_median_scale_does_not_mutate_input():
    pred = np.array([[10.0, 20.0], [30.0, 40.0]])
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred_before = pred.copy()
    gt_before = gt.copy()
    aligned, _ = median_scale(pred, gt)
    assert np.array_equal(pred, pred_before)
    assert np.array_equal(gt, gt_before)
    assert aligned is not pred
    assert not np.shares_memory(aligned, pred)


def test_median_scale_with_valid_mask():
    pred = np.array([[10.0, 100.0], [30.0, 40.0]])
    gt = np.array([[1.0, 1.0], [3.0, 4.0]])
    valid = np.array([[True, False], [True, True]])
    aligned, scale = median_scale(pred, gt, valid)
    # valid pred medians [10, 40]? no: valid pixels (0,0),(1,0),(1,1)
    # medians: pred [10,30,40] -> 30, gt [1,3,4] -> 3, scale = 0.1
    assert scale == pytest.approx(0.1)
    assert np.allclose(aligned[0, 1], 100.0 * scale)  # invalid pixel still scaled but flagged


# 12. empty valid mask
def test_empty_valid_mask_raises():
    gt = np.zeros((2, 2))
    with pytest.raises(ValueError, match="no valid depth pixels"):
        rmse(PRED_ZERO, gt)
    valid = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="no valid depth pixels"):
        rmse(PRED_ZERO, GT, valid)


# 13. shape mismatch
def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same spatial shape"):
        rmse(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same spatial shape"):
        evaluate_depth(np.zeros((2, 3)), np.zeros((2, 2)), align=None)


def test_valid_mask_shape_mismatch_raises():
    with pytest.raises(ValueError, match="valid mask"):
        rmse(PRED_ZERO, GT, valid=np.ones((3, 3), dtype=bool))


# 14. non-finite values
def test_all_nonfinite_gt_raises():
    gt = np.full((2, 2), np.nan)
    with pytest.raises(ValueError, match="no valid depth pixels"):
        mae(PRED_ZERO, gt)


def test_all_nonfinite_pred_raises():
    with pytest.raises(ValueError, match="no valid depth pixels"):
        mae(np.full((2, 2), np.nan), GT)


# 15. zero/negative GT and non-positive prediction handling
def test_nonpositive_prediction_rejected_for_delta():
    with pytest.raises(ValueError, match="positive prediction"):
        delta_accuracy(PRED_ZERO, GT)
    with pytest.raises(ValueError, match="positive prediction"):
        median_scale(PRED_ZERO, GT)


# 16. aggregate evaluate_depth()
def test_evaluate_depth_aggregate_relative():
    result = evaluate_depth(PRED4, GT4, align=None)
    assert set(result) == {"rmse", "mae", "abs_rel", "delta1", "delta2", "delta3"}
    assert result["rmse"] == pytest.approx(np.sqrt(19.5))
    assert result["mae"] == pytest.approx(3.5)
    assert result["abs_rel"] == pytest.approx(0.35)
    assert result["delta1"] == pytest.approx(0.5)
    assert result["delta2"] == pytest.approx(0.5)
    assert result["delta3"] == pytest.approx(0.75)


def test_evaluate_depth_rejects_unknown_align():
    with pytest.raises(ValueError, match="align"):
        evaluate_depth(PRED4, GT4, align="lsq")


# 17. NumPy input
def test_numpy_input():
    result = evaluate_depth(PRED4, GT4, align=None)
    assert isinstance(result["rmse"], float)


# 18. Torch input
def test_torch_input_matches_numpy():
    numpy_result = evaluate_depth(PRED4, GT4, align=None)
    torch_result = evaluate_depth(torch.tensor(PRED4), torch.tensor(GT4), align=None)
    for key in numpy_result:
        assert torch_result[key] == pytest.approx(numpy_result[key])


def test_mixed_torch_numpy_input():
    pred = GT4 * 2.0
    result = evaluate_depth(torch.tensor(pred), GT4, align="median")
    assert result["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert result["scale"] == pytest.approx(0.5)


# 19. perfect prediction
def test_perfect_prediction():
    result = evaluate_depth(GT4, GT4, align=None)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["abs_rel"] == pytest.approx(0.0)
    assert result["delta1"] == pytest.approx(1.0)
    assert result["delta2"] == pytest.approx(1.0)
    assert result["delta3"] == pytest.approx(1.0)


# 20. constant-scale prediction corrected by median scaling
def test_constant_scale_prediction_corrected_by_median_scaling():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = 2.0 * gt
    aligned = evaluate_depth(pred, gt, align="median")
    assert aligned["scale"] == pytest.approx(0.5)
    assert aligned["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert aligned["delta1"] == pytest.approx(1.0)

    raw = evaluate_depth(pred, gt, align=None)
    assert raw["rmse"] > 1.0
    assert "scale" not in raw


# extra: delta thresholds validation
def test_delta_accuracy_validates_thresholds():
    with pytest.raises(ValueError, match="threshold"):
        delta_accuracy(PRED4, GT4, thresholds=(0.9,))
    with pytest.raises(ValueError, match="threshold"):
        delta_accuracy(PRED4, GT4, thresholds=())


def test_delta_accuracy_custom_threshold():
    deltas = delta_accuracy(PRED4, GT4, thresholds=(1.25, 1.9))
    assert len(deltas) == 2
    assert deltas[0] == pytest.approx(0.5)
    # ratios [2.0, 1.2, 1.0, 1.7] -> pass 1.9: 1.2, 1.0 and 1.7 -> 0.75
    assert deltas[1] == pytest.approx(0.75)