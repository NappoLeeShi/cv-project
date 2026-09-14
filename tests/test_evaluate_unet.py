import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from evaluation.evaluate_unet import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    build_dataset,
    build_parser,
    build_results,
    evaluate_dataset,
    load_checkpoint,
    write_results,
)
from evaluation.segmentation_metrics import evaluate_segmentation
from preprocessing.cityscapes import TRAIN_ID_CLASS_NAMES


class TinySegDataset(Dataset):
    """Synthetic ``{image, label, image_path}`` pairs in the project dict format.

    The image is a ``[C, H, W]`` one-hot-like logits tensor whose argmax equals
    the ground-truth class for every valid pixel, so an identity model scores a
    perfect pixel accuracy over the non-``255`` pixels.
    """

    def __init__(self, num_pairs=4, num_classes=3, image_size=(16, 32), seed=0):
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.pairs = []
        for _ in range(num_pairs):
            label = rng.integers(0, num_classes, size=image_size).astype(np.int64)
            ignore = rng.random(image_size) < 0.15
            label[ignore] = 255
            label_tensor = torch.from_numpy(label)
            image = torch.zeros((num_classes, *image_size))
            for c in range(num_classes):
                image[c][label_tensor == c] = 1.0
            self.pairs.append({"image": image, "label": label_tensor})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return {
            "image": item["image"],
            "label": item["label"],
            "image_path": f"mock/city_{idx}_leftImg8bit.png",
        }


class ConvProbe(nn.Module):
    """Tiny parameterised model used to assert no-grad / weights unchanged."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _concatenated_maps(dataset: TinySegDataset) -> tuple[torch.Tensor, torch.Tensor]:
    preds, targets = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        preds.append(item["image"].argmax(dim=0))
        targets.append(item["label"])
    return torch.cat(preds, dim=1), torch.cat(targets, dim=1)


# ------------------------------------------------------------------ checkpoint
def test_load_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "ckpt.pth"
    torch.save(
        {
            "model_state_dict": {"w": torch.zeros(2, 3)},
            "epoch": 20,
            "num_classes": 19,
            "ignore_index": 255,
            "val_miou": 0.42,
        },
        path,
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint["epoch"] == 20
    assert checkpoint["num_classes"] == 19
    assert checkpoint["ignore_index"] == 255
    assert set(checkpoint.keys()) >= {"model_state_dict", "epoch", "num_classes"}


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_checkpoint(tmp_path / "missing.pth")


def test_load_checkpoint_rejects_bad_format(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"optimizer": "state"}, path)
    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path)


# ------------------------------------------------------------------ evaluation
def test_evaluate_dataset_perfect_prediction_metrics():
    dataset = TinySegDataset(num_pairs=4, num_classes=3)
    metrics = evaluate_dataset(
        nn.Identity(), dataset, device="cpu", num_classes=3, ignore_index=255, batch_size=1
    )
    assert metrics["num_samples"] == 4
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["mean_dice"] == pytest.approx(1.0)
    assert np.allclose(metrics["iou_per_class"], 1.0)
    assert np.allclose(metrics["dice_per_class"], 1.0)


def test_evaluate_dataset_matches_reference_metrics():
    dataset = TinySegDataset(num_pairs=6, num_classes=4)
    metrics = evaluate_dataset(
        nn.Identity(), dataset, device="cpu", num_classes=4, ignore_index=255
    )
    pred, target = _concatenated_maps(dataset)
    reference = evaluate_segmentation(pred, target, num_classes=4, ignore_index=255)
    assert metrics["pixel_accuracy"] == pytest.approx(reference["pixel_accuracy"])
    assert metrics["miou"] == pytest.approx(reference["miou"])
    assert metrics["mean_dice"] == pytest.approx(reference["mean_dice"])
    assert np.allclose(metrics["iou_per_class"], reference["iou_per_class"])
    assert np.allclose(metrics["dice_per_class"], reference["dice_per_class"])
    assert np.allclose(metrics["class_accuracy"], reference["class_accuracy"])


def test_evaluate_dataset_batch_size_invariant():
    dataset = TinySegDataset(num_pairs=8, num_classes=3)
    single = evaluate_dataset(
        nn.Identity(), dataset, device="cpu", num_classes=3, ignore_index=255, batch_size=1
    )
    batched = evaluate_dataset(
        nn.Identity(), dataset, device="cpu", num_classes=3, ignore_index=255, batch_size=4
    )
    assert single["num_samples"] == batched["num_samples"] == 8
    for key in ("pixel_accuracy", "miou", "mean_dice"):
        assert single[key] == pytest.approx(batched[key])
    assert np.allclose(single["iou_per_class"], batched["iou_per_class"])


def test_evaluate_dataset_no_grad_and_weights_unchanged():
    model = ConvProbe(num_classes=3)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    evaluate_dataset(
        model, TinySegDataset(num_pairs=4, num_classes=3), device="cpu", num_classes=3
    )
    for name, param in model.named_parameters():
        assert torch.equal(before[name], param.detach()), "weights must not change"
        assert param.grad is None, "no gradients may be produced during evaluation"


def test_evaluate_dataset_output_dimensions_and_shapes():
    dataset = TinySegDataset(num_pairs=4, num_classes=3, image_size=(16, 32))
    metrics = evaluate_dataset(
        nn.Identity(), dataset, device="cpu", num_classes=3, ignore_index=255
    )
    assert len(metrics["iou_per_class"]) == 3
    assert len(metrics["dice_per_class"]) == 3
    assert len(metrics["class_accuracy"]) == 3


def test_evaluate_dataset_saves_prediction_masks(tmp_path):
    dataset = TinySegDataset(num_pairs=2, num_classes=3, image_size=(10, 20))
    save_dir = tmp_path / "preds"
    evaluate_dataset(
        nn.Identity(),
        dataset,
        device="cpu",
        num_classes=3,
        ignore_index=255,
        save_predictions_dir=save_dir,
    )
    saved = sorted(save_dir.iterdir())
    assert len(saved) == 2
    assert saved[0].name == f"{Path(dataset[0]['image_path']).stem}_pred.png"
    array = np.asarray(Image.open(saved[0]).convert("L"))
    assert tuple(array.shape) == (10, 20)
    assert set(np.unique(array)) <= {0, 1, 2}


def test_evaluate_dataset_ignores_255_pixels():
    dataset = TinySegDataset(num_pairs=1, num_classes=2, seed=3)
    item = dataset[0]
    valid = item["label"] != 255
    expected = float((item["image"].argmax(dim=0)[valid] == item["label"][valid]).float().mean())
    metrics = evaluate_dataset(
        nn.Identity(), dataset, device="cpu", num_classes=2, ignore_index=255
    )
    assert metrics["pixel_accuracy"] == pytest.approx(expected)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------- dataset construction
def test_build_dataset_fails_fast_without_data(tmp_path, monkeypatch):
    from utils import config as config_utils

    monkeypatch.setattr(config_utils, "ROOT_DIR", tmp_path)
    cfg = {
        "data": {"cityscapes_root": str(tmp_path / "data" / "cityscapes"), "image_size": [256, 512]},
        "model": {"num_classes": 19},
    }
    with pytest.raises(FileNotFoundError, match="not found"):
        build_dataset(cfg)


# ------------------------------------------------------------ results / JSON
def _fake_metrics(num_classes=19, num_samples=500):
    return {
        "num_samples": num_samples,
        "pixel_accuracy": 0.9021,
        "miou": 0.4262,
        "mean_dice": 0.55,
        "iou_per_class": np.linspace(0.1, 0.8, num_classes),
        "dice_per_class": np.linspace(0.2, 0.9, num_classes),
        "class_accuracy": np.linspace(0.3, 1.0, num_classes),
    }


def test_build_results_structure():
    config = {"model": {"num_classes": 19}, "data": {"image_size": [256, 512]}}
    checkpoint = {
        "epoch": 20,
        "num_classes": 19,
        "ignore_index": 255,
        "val_miou": 0.4262,
        "val_pixel_accuracy": 0.9021,
    }
    results = build_results(
        config, checkpoint, "checkpoints/unet_cityscapes.pth", _fake_metrics()
    )
    for key in (
        "checkpoint", "split", "num_samples", "num_classes", "ignore_index",
        "image_size", "checkpoint_epoch", "pixel_accuracy", "miou",
        "mean_dice", "per_class",
    ):
        assert key in results
    assert results["num_samples"] == 500
    assert results["num_classes"] == 19
    assert results["ignore_index"] == 255
    assert results["image_size"] == [256, 512]
    assert results["checkpoint_epoch"] == 20
    assert list(results["per_class"]) == TRAIN_ID_CLASS_NAMES
    entry = results["per_class"]["road"]
    assert set(entry) == {"iou", "dice", "accuracy"}


def test_build_results_rejects_mismatched_num_classes():
    config = {"model": {"num_classes": 19}, "data": {"image_size": [256, 512]}}
    checkpoint = {"num_classes": 8, "ignore_index": 255}
    with pytest.raises(ValueError, match="config declares 19"):
        build_results(config, checkpoint, "x.pth", _fake_metrics())


def test_build_results_rejects_mismatched_ignore_index():
    config = {"model": {"num_classes": 19}, "data": {"image_size": [256, 512]}}
    checkpoint = {"num_classes": 19, "ignore_index": 254}
    with pytest.raises(ValueError, match="ignore_index"):
        build_results(config, checkpoint, "x.pth", _fake_metrics())


def test_write_results_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "results.json"
    results = {"miou": 0.5, "per_class": {"road": {"iou": 0.9}}}
    written = write_results(results, path)
    assert written == path
    with path.open("r", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    assert reloaded == results


# ------------------------------------------------------------------------ CLI
def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.config == "unet"
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.output == DEFAULT_OUTPUT
    assert args.split == "val"
    assert args.device == "auto"
    assert args.batch_size == 1
    assert args.save_predictions is False


def test_build_parser_save_predictions_flag():
    args = build_parser().parse_args(["--save-predictions", "--split", "val"])
    assert args.save_predictions is True
    args = build_parser().parse_args(["--checkpoint", "ckpt.pth", "--output", "o.json"])
    assert args.checkpoint == "ckpt.pth"
    assert args.output == "o.json"