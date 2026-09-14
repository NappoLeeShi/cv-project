import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from evaluation.depth_metrics import evaluate_depth
from evaluation.evaluate_midas import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    build_dataset,
    build_parser,
    build_results,
    evaluate_checkpoint,
    evaluate_dataset,
    load_checkpoint,
    write_results,
)
from models.midas.model import MiDaSModel


class ConstantDepth(nn.Module):
    """Deterministic stub: returns a scalar depth value for every input pixel."""

    def __init__(self, value: float = 2.0) -> None:
        super().__init__()
        self.value = value
        self._scale = nn.Parameter(torch.tensor(value))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3]),
            self.value,
            dtype=torch.float32,
        )


def _make_predictor(value: float = 2.0, input_size: int = 64):
    """Build a MidDepthPredictor backed by ConstantDepth."""
    from models.midas.inference import MidDepthPredictor

    return MidDepthPredictor(
        model=ConstantDepth(value),
        device="cpu",
        input_size=input_size,
        use_official_transform=False,
    )


# ---------------------------------------------------------------- checkpoint
def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_checkpoint(tmp_path / "missing.pth")


def test_load_checkpoint_success(tmp_path, monkeypatch):
    stub = ConstantDepth(5.0)
    monkeypatch.setattr("evaluation.evaluate_midas._build_backend", lambda dev: stub)
    ck_path = tmp_path / "fake_midas.pth"
    torch.save(stub.state_dict(), ck_path)
    model = load_checkpoint(ck_path, device="cpu")
    assert isinstance(model, MiDaSModel)
    assert model.training is False


def test_load_checkpoint_loads_weights(tmp_path, monkeypatch):
    stub = ConstantDepth(5.0)
    monkeypatch.setattr("evaluation.evaluate_midas._build_backend", lambda dev: stub)
    ck_path = tmp_path / "fake_midas.pth"
    state = stub.state_dict()
    state["_scale"] = torch.tensor(12.345)
    torch.save(state, ck_path)
    loaded = load_checkpoint(ck_path, device="cpu")
    params = dict(loaded.named_parameters())
    scale_val = params["backend._scale"]
    assert scale_val.item() == pytest.approx(12.345)


# ---------------------------------------------------------------- model mode
def test_predictor_model_is_in_eval_mode(tmp_path, monkeypatch):
    stub = ConstantDepth(5.0)
    monkeypatch.setattr("evaluation.evaluate_midas._build_backend", lambda dev: stub)
    ck_path = tmp_path / "fake_midas.pth"
    torch.save(stub.state_dict(), ck_path)
    model = load_checkpoint(ck_path, device="cpu")
    predictor_model = model
    assert predictor_model.training is False


# ---------------------------------------------------------------- no-grad
def test_evaluate_dataset_no_grad_and_weights_unchanged(tmp_path):
    from preprocessing.kitti import KittiDepthDataset
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    _make_frame(tmp_path, "0000000002", 12345, img_size=(16, 8))
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=1000.0)
    stub = ConstantDepth(5.0)
    model = MiDaSModel(backend=stub, variant="dpt_hybrid")
    before = {k: v.detach().clone() for k, v in stub.named_parameters()}
    evaluate_dataset(model, ds, device="cpu", input_size=64, align="median")
    for name, param in stub.named_parameters():
        assert torch.equal(before[name], param.detach()), f"weights changed: {name}"
        assert param.grad is None, f"gradient produced: {name}"


# ---------------------------------------------------------------- prediction shape
def test_prediction_shape_matches_gt(tmp_path):
    from preprocessing.kitti import KittiDepthDataset
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=1000.0)
    sample = ds[0]
    gt = np.asarray(sample["depth"], dtype=np.float64)
    predictor = _make_predictor(5.0)
    pred = predictor.predict(Image.open(sample["image_path"]).convert("RGB"))
    assert pred.numpy().shape == gt.shape


# ---------------------------------------------------------------- valid mask
def test_valid_depth_mask_respected(tmp_path):
    from preprocessing.kitti import KittiDepthDataset
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=1000.0)
    sample = ds[0]
    gt = np.asarray(sample["depth"], dtype=np.float64)
    valid = np.asarray(sample["valid_mask"], dtype=bool)
    # Constant pred=2.0, GT=5.0 → raw error
    pred = np.full_like(gt, 2.0)
    result = evaluate_depth(pred, gt, valid, align=None)
    assert result["rmse"] == pytest.approx(3.0, abs=1e-6)
    assert result["abs_rel"] == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------- median scaling
def test_median_scaling_behavior(tmp_path):
    from preprocessing.kitti import KittiDepthDataset
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=1000.0)
    sample = ds[0]
    gt = np.asarray(sample["depth"], dtype=np.float64)
    valid = np.asarray(sample["valid_mask"], dtype=bool)
    # pred=2.5, GT=5.0 → scale = 5.0/2.5 = 2.0, aligned=5.0
    pred = np.full_like(gt, 2.5)
    result = evaluate_depth(pred, gt, valid, align="median")
    assert result["scale"] == pytest.approx(2.0)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert result["delta1"] == pytest.approx(1.0)


# ---------------------------------------------------------------- metric keys
def test_evaluate_depth_returns_expected_keys():
    gt = np.array([[3.0, 3.0], [3.0, 3.0]])
    pred = np.array([[2.0, 2.0], [2.0, 2.0]])
    result = evaluate_depth(pred, gt, align=None)
    assert set(result) == {"rmse", "mae", "abs_rel", "delta1", "delta2", "delta3"}


# ---------------------------------------------------------------- evaluate_dataset
def test_evaluate_dataset_returns_expected_structure(tmp_path):
    from preprocessing.kitti import KittiDepthDataset
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    _make_frame(tmp_path, "0000000002", 12345, img_size=(16, 8))
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=1000.0)
    stub = ConstantDepth(5.0)
    model = MiDaSModel(backend=stub, variant="dpt_hybrid")
    metrics = evaluate_dataset(model, ds, device="cpu", input_size=64, align="median")
    assert metrics["num_samples"] == 2
    assert metrics["skipped"] == 0
    for key in ("rmse", "mae", "abs_rel", "delta1", "delta2", "delta3"):
        assert key in metrics
        assert metrics[key] >= 0.0
    assert metrics["mean_scale"] is not None
    assert metrics["mean_scale"] > 0.0


def test_evaluate_dataset_perfect_when_prediction_constant(tmp_path):
    from preprocessing.kitti import KittiDepthDataset
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    _make_frame(tmp_path, "0000000002", 12345, img_size=(16, 8))
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=1000.0)
    stub = ConstantDepth(5.0)
    model = MiDaSModel(backend=stub, variant="dpt_hybrid")
    metrics = evaluate_dataset(model, ds, device="cpu", input_size=64, align="median")
    assert metrics["num_samples"] == 2
    # frame1: GT=5.0, pred=5.0 → scale=1.0, aligned=5.0 → perfect
    # frame2: GT=12.345, pred=5.0 → scale=2.469, aligned=12.345 → perfect
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["mean_scale"] is not None
    assert metrics["mean_scale"] > 0


def test_evaluate_dataset_saves_predictions(tmp_path):
    from preprocessing.kitti import KittiDepthDataset
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    _make_frame(tmp_path, "0000000002", 12345, img_size=(16, 8))
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=1000.0)
    stub = ConstantDepth(5.0)
    model = MiDaSModel(backend=stub, variant="dpt_hybrid")
    save_dir = tmp_path / "preds"
    evaluate_dataset(
        model, ds, device="cpu", input_size=64, align="median",
        save_predictions_dir=save_dir,
    )
    saved = sorted(save_dir.iterdir())
    assert len(saved) == 2
    assert saved[0].suffix == ".npy"
    arr = np.load(saved[0])
    assert arr.dtype == np.float32
    assert arr.ndim == 2


def test_evaluate_dataset_empty_raises():
    from unittest.mock import MagicMock

    stub = ConstantDepth(5.0)
    model = MiDaSModel(backend=stub, variant="dpt_hybrid")
    empty_ds = MagicMock()
    empty_ds.__len__ = MagicMock(return_value=0)
    with pytest.raises(RuntimeError, match="no samples"):
        evaluate_dataset(model, empty_ds, device="cpu", input_size=64)


# ---------------------------------------------------------------- build_results
def test_build_results_structure():
    metrics = {
        "rmse": 1.23,
        "mae": 0.45,
        "abs_rel": 0.12,
        "delta1": 0.9,
        "delta2": 0.8,
        "delta3": 0.7,
        "mean_scale": 0.5,
        "num_samples": 1000,
        "skipped": 0,
    }
    results = build_results(
        checkpoint_path="checkpoints/dpt_large_384.pt",
        dataset_len=1000,
        metrics=metrics,
        split="val",
        align="median",
        config={"eval": {"depth_cap_m": 80}},
    )
    assert results["model"] == "MiDaS DPT-Large"
    assert results["num_samples"] == 1000
    assert results["num_expected"] == 1000
    assert results["skipped"] == 0
    assert results["relative_inverse_depth"] is True
    assert results["larger_value_is_closer"] is True
    assert results["median_scaling"] is True
    assert results["depth_cap_m"] == 80
    assert results["metrics"]["rmse"] == pytest.approx(1.23)
    assert results["metrics"]["absrel"] == pytest.approx(0.12)
    assert results["mean_scale"] == pytest.approx(0.5)


def test_build_results_no_alignment():
    metrics = {
        "rmse": 2.0, "mae": 1.0, "abs_rel": 0.5,
        "delta1": 0.6, "delta2": 0.5, "delta3": 0.4,
        "mean_scale": None, "num_samples": 100, "skipped": 0,
    }
    results = build_results(
        checkpoint_path="x", dataset_len=100, metrics=metrics,
        split="val", align=None, config=None,
    )
    assert results["median_scaling"] is False
    assert results["mean_scale"] is None


# ---------------------------------------------------------------- JSON roundtrip
def test_write_results_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "result.json"
    results = {"model": "MiDaS", "metrics": {"rmse": 1.0}, "mean_scale": 0.5}
    written = write_results(results, path)
    assert written == path
    with path.open("r", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    assert reloaded == results


# ---------------------------------------------------------------- CLI
def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.config == "midas"
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.output == DEFAULT_OUTPUT
    assert args.split == "val"
    assert args.device == "auto"
    assert args.batch_size == 1
    assert args.save_predictions is False
    assert args.align is None


def test_build_parser_save_predictions_flag():
    args = build_parser().parse_args([
        "--save-predictions", "--align", "none",
        "--checkpoint", "ckpt.pth", "--output", "o.json",
    ])
    assert args.save_predictions is True
    assert args.align == "none"
    assert args.checkpoint == "ckpt.pth"
    assert args.output == "o.json"


# ---------------------------------------------------------------- integration
def test_evaluate_checkpoint_integration(tmp_path, monkeypatch):
    from tests.test_kitti import _make_frame

    _make_frame(tmp_path, "0000000001", 5000, img_size=(16, 8))
    _make_frame(tmp_path, "0000000002", 12345, img_size=(16, 8))
    cfg = {
        "data": {"kitti": {"root": str(tmp_path)}, "num_workers": 0},
        "depth": {"metric_alignment": "median"},
        "eval": {"depth_cap_m": None},
        "preprocessing": {"input_size": 64},
        "env": {"seed": 42},
    }
    stub = ConstantDepth(5.0)
    monkeypatch.setattr(
        "evaluation.evaluate_midas.load_checkpoint",
        lambda path, device="cpu": MiDaSModel(
            backend=stub, variant="dpt_hybrid"
        ).eval(),
    )
    ck_path = tmp_path / "fake.pth"
    torch.save({}, ck_path)
    results = evaluate_checkpoint(
        cfg, ck_path, device="cpu", split="val",
    )
    assert results["num_samples"] == 2
    assert results["metrics"]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert results["metrics"]["delta1"] == pytest.approx(1.0)
    assert results["relative_inverse_depth"] is True
    assert results["median_scaling"] is True
