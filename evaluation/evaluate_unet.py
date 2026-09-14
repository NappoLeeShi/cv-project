"""Step 13 — evaluate the trained U-Net on the full Cityscapes validation split.

Runs the existing U-Net checkpoint over every Cityscapes validation image using
the project dataset loader, aggregates one global confusion matrix
(``ignore_index=255`` excluded) and reports the existing segmentation metrics.

This is a pure evaluation step: no gradients, no parameter updates, no training.

CLI::

    python -m evaluation.evaluate_unet
    python -m evaluation.evaluate_unet --config unet \\
        --checkpoint checkpoints/unet_cityscapes.pth

Results are written to ``outputs/analysis/unet_cityscapes_evaluation.json``
(override with ``--output``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from evaluation.segmentation_metrics import aggregate_from_confusion_matrix, confusion_matrix
from models.unet.inference import load_model
from models.unet.model import UNet
from preprocessing.cityscapes import IGNORE_INDEX, CityscapesDataset, TRAIN_ID_CLASS_NAMES
from utils.config import ConfigError, get, load_config, resolve_path
from utils.device import resolve_device
from utils.logger import get_logger, setup_logging
from utils.seed import set_seed

LOGGER = get_logger("evaluation.evaluate_unet")

DEFAULT_CHECKPOINT = "checkpoints/unet_cityscapes.pth"
DEFAULT_OUTPUT = "outputs/analysis/unet_cityscapes_evaluation.json"
DEFAULT_SPLIT = "val"
DEFAULT_NUM_CLASSES = 19


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.evaluate_unet",
        description="Evaluate the trained U-Net on the Cityscapes validation split (no training).",
    )
    parser.add_argument(
        "--config",
        default="unet",
        help="Config name (e.g. 'unet').",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to the U-Net checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination JSON file for the results (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to evaluate, e.g. 'val' (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' (default), 'cpu' or 'cuda'.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also save per-image class-id prediction masks under "
        "outputs/segmentation/unet_cityscapes/.",
    )
    return parser


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a U-Net checkpoint dict (``model_state_dict`` + metadata).

    Raises ``FileNotFoundError`` when the file is missing and ``ValueError`` when
    the file does not contain a ``model_state_dict`` mapping.
    """
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=resolve_device(device), weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unrecognized checkpoint format in {checkpoint_path}: expected a dict "
            f"containing a 'model_state_dict' key."
        )
    return checkpoint


def _task_params(config: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[int, int]:
    """Resolve and cross-check ``num_classes`` and ``ignore_index``.

    The config defines the task layout; the checkpoint metadata (when present)
    must agree, so a checkpoint trained for a different task can never be
    silently scored.
    """
    num_classes = int(get(config, "model.num_classes", DEFAULT_NUM_CLASSES))
    ignore_index = int(IGNORE_INDEX)
    if checkpoint is not None:
        ck_classes = checkpoint.get("num_classes")
        ck_ignore = checkpoint.get("ignore_index")
        if ck_classes is not None and int(ck_classes) != num_classes:
            raise ValueError(
                f"Checkpoint was trained for {int(ck_classes)} classes but the "
                f"config declares {num_classes}."
            )
        if ck_ignore is not None and int(ck_ignore) != ignore_index:
            raise ValueError(
                f"Checkpoint ignore_index={int(ck_ignore)} disagrees with the "
                f"Cityscapes ignore_index={ignore_index}."
            )
    return num_classes, ignore_index


def build_dataset(config: dict[str, Any], split: str = DEFAULT_SPLIT) -> CityscapesDataset:
    """Build the Cityscapes dataset for ``split`` from the U-Net config."""
    root = resolve_path(get(config, "data.cityscapes_root", "data/cityscapes"))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Cityscapes dataset not found at {root}. Evaluation never downloads "
            f"data; obtain the dataset manually first."
        )
    image_size = get(config, "data.image_size", (256, 512))
    split_dir = get(config, "data.split_dir", {}) or {}
    return CityscapesDataset(
        root=root,
        split=split_dir.get(split, split),
        image_dir=get(config, "data.image_dir"),
        label_dir=get(config, "data.label_dir"),
        image_size=image_size,
        max_samples=None,
    )


def _save_prediction_mask(prediction: torch.Tensor, source_path: str, save_dir: Path) -> Path:
    stem = Path(source_path).stem
    target = save_dir / f"{stem}_pred.png"
    mask = prediction.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(mask, mode="L").save(target)
    return target


def evaluate_dataset(
    model: torch.nn.Module,
    dataset: CityscapesDataset,
    device: str | torch.device,
    num_classes: int = DEFAULT_NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    batch_size: int = 1,
    num_workers: int = 0,
    save_predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run ``model`` over ``dataset`` and aggregate metrics from one global matrix.

    The model is never trained: predictions are produced under ``torch.no_grad()``
    and compared with the ground-truth labels (trainId 0..18, 255 = ignore).
    Returns ``num_samples`` plus the same key set as
    :func:`evaluation.segmentation_metrics.evaluate_segmentation`.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=max(int(num_workers), 0),
    )

    save_dir = (
        resolve_path(save_predictions_dir) if save_predictions_dir is not None else None
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    total_cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    num_samples = 0
    for batch in loader:
        images = batch["image"]
        labels = batch["label"]
        if labels is None:
            raise ValueError(
                "Evaluation requires ground-truth labels for every image."
            )
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
        predictions = logits.argmax(dim=1)

        cm = confusion_matrix(
            predictions.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        total_cm += cm
        num_samples += int(images.shape[0])

        if save_dir is not None:
            for i in range(int(images.shape[0])):
                _save_prediction_mask(predictions[i], batch["image_path"][i], save_dir)

    if num_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples (empty dataset?).")

    metrics = aggregate_from_confusion_matrix(total_cm)
    return {"num_samples": num_samples, **metrics}


def build_results(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
    metrics: dict[str, Any],
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Assemble the reproducible JSON result dict (per-class entries by name)."""
    num_classes, ignore_index = _task_params(config, checkpoint)

    per_class: dict[str, dict[str, float]] = {}
    for class_id in range(num_classes):
        name = TRAIN_ID_CLASS_NAMES[class_id] if class_id < len(TRAIN_ID_CLASS_NAMES) else str(class_id)
        per_class[name] = {
            "iou": float(metrics["iou_per_class"][class_id]),
            "dice": float(metrics["dice_per_class"][class_id]),
            "accuracy": float(metrics["class_accuracy"][class_id]),
        }

    image_size = list(get(config, "data.image_size", (256, 512)))
    checkpoint_epoch = checkpoint.get("epoch")
    if checkpoint_epoch is None:
        checkpoint_epoch = checkpoint.get("best_epoch")

    return {
        "checkpoint": str(resolve_path(checkpoint_path)),
        "split": split,
        "num_samples": int(metrics["num_samples"]),
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "checkpoint_epoch": (
            int(checkpoint_epoch) if checkpoint_epoch is not None else None
        ),
        "training_val_miou": (
            float(checkpoint["val_miou"]) if checkpoint.get("val_miou") is not None else None
        ),
        "training_val_pixel_accuracy": (
            float(checkpoint["val_pixel_accuracy"])
            if checkpoint.get("val_pixel_accuracy") is not None
            else None
        ),
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
        "miou": float(metrics["miou"]),
        "mean_dice": float(metrics["mean_dice"]),
        "per_class": per_class,
    }


def write_results(results: dict[str, Any], output_path: str | Path) -> Path:
    """Persist the results dict as readable JSON."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return path


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    output_path: str | Path | None = None,
    save_predictions: bool = False,
    split: str = DEFAULT_SPLIT,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a saved U-Net checkpoint on the full Cityscapes split.

    Loads the checkpoint metadata, constructs the U-Net and its weights, builds
    the dataset, aggregates the metrics and optionally persists the JSON results.
    """
    if seed is not None:
        set_seed(seed)

    device_obj = resolve_device(device)
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device=device_obj)
    num_classes, ignore_index = _task_params(config, checkpoint)

    model = UNet.from_config(config)
    load_model(model, checkpoint_path, device=str(device_obj))

    dataset = build_dataset(config, split=split)
    LOGGER.info(
        "Evaluating %d %s samples with %s (%s classes, ignore_index=%d, device=%s)",
        len(dataset),
        split,
        checkpoint_path,
        num_classes,
        ignore_index,
        device_obj,
    )

    save_dir = (
        resolve_path("outputs/segmentation/unet_cityscapes") if save_predictions else None
    )
    metrics = evaluate_dataset(
        model,
        dataset,
        device=device_obj,
        num_classes=num_classes,
        ignore_index=ignore_index,
        batch_size=batch_size,
        num_workers=num_workers,
        save_predictions_dir=save_dir,
    )

    results = build_results(
        config, checkpoint, checkpoint_path, metrics, split=split
    )
    if output_path is not None:
        write_results(results, output_path)
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level="INFO")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    seed = int(get(config, "env.seed", 42))
    num_workers = int(get(config, "data.num_workers", 0))
    batch_size = int(args.batch_size if args.batch_size is not None else 1)

    try:
        results = evaluate_checkpoint(
            config,
            args.checkpoint,
            device=args.device,
            batch_size=batch_size,
            num_workers=num_workers,
            output_path=args.output,
            save_predictions=args.save_predictions,
            split=args.split,
            seed=seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path = write_results(results, args.output)  # ensure fresh copy
    ious = {name: entry["iou"] for name, entry in results["per_class"].items()}
    best = max(ious, key=ious.get)
    worst = min(ious, key=ious.get)

    print(f"Evaluated {results['num_samples']} Cityscapes {args.split} images")
    print(f"Pixel Accuracy : {results['pixel_accuracy']:.6f}")
    print(f"mIoU           : {results['miou']:.6f}")
    print(f"Mean Dice      : {results['mean_dice']:.6f}")
    print(f"Best class IoU : {best} ({ious[best]:.4f})")
    print(f"Worst class IoU: {worst} ({ious[worst]:.4f})")
    print(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())