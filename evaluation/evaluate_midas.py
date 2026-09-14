"""Step 14 — evaluate the pretrained MiDaS DPT-Large on the full KITTI validation split.

Runs the existing MiDaS DPT-Large checkpoint over every KITTI validation image
using the project's KITTI depth loader and MiDaS inference wrapper, computes
depth metrics with per-image median scaling, and reports the aggregated results.

This is an evaluation step only: no gradients, no parameter updates, no training.

The MiDaS model produces *relative inverse depth* (larger value = closer to the
camera). Median scaling is applied at evaluation time only, using
``scale = median(gt_valid) / median(pred_valid)`` to align the relative
prediction with the ground-truth metric range before computing depth errors.
MiDaS itself never predicts metric depth.

CLI::

    python -m evaluation.evaluate_midas
    python -m evaluation.evaluate_midas \\
        --config midas --checkpoint checkpoints/dpt_large_384.pt

Results are saved to ``outputs/analysis/midas_kitti_evaluation.json``
(override with ``--output``).

Hub-name note
-------------
The upstream ``intel-isl/MiDaS`` torch.hub now exports ``DPT_Large``
(PascalCase), but the project's ``models.midas.model.validate_variant``
normalises the variant name to lowercase ``"dpt_large"`` for all hub calls,
which no longer resolves in the current hubconf. The architecture and weight
keys are identical between the upstream ``DPT_Large`` entry point and the
project checkpoint, so this module builds the raw backend directly via
``torch.hub.load(MODEL_SOURCE, "DPT_Large", pretrained=False)`` and wraps it
with the existing ``MiDaSModel`` and ``MidDepthPredictor``; the checkpoint is
loaded using the existing ``_load_weights`` helper — no model loading logic
is duplicated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from evaluation.depth_metrics import evaluate_depth
from models.midas.inference import MidDepthPredictor
from models.midas.model import (
    MODEL_SOURCE,
    MiDaSModel,
    MiDaSError,
    _load_weights,
)
from preprocessing.kitti import KittiDepthDataset
from utils.config import ConfigError, get, load_config, resolve_path
from utils.device import resolve_device
from utils.logger import get_logger, setup_logging
from utils.seed import set_seed

LOGGER = get_logger("evaluation.evaluate_midas")

DEFAULT_CHECKPOINT = "checkpoints/dpt_large_384.pt"
DEFAULT_OUTPUT = "outputs/analysis/midas_kitti_evaluation.json"
DEFAULT_SPLIT = "val"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.evaluate_midas",
        description=(
            "Evaluate the pretrained MiDaS DPT-Large on the KITTI "
            "validation split (no training)."
        ),
    )
    parser.add_argument(
        "--config",
        default="midas",
        help="Config name (e.g. 'midas').",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to the MiDaS DPT-Large checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination JSON file for the results (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' (default), 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--align",
        default=None,
        choices=["median", "none"],
        help="Evaluation alignment mode (default from config: 'median').",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted depth maps under outputs/depth/midas_kitti/.",
    )
    return parser


# ------------------------------------------------------------ model loading
def _build_backend(device: str | torch.device) -> torch.nn.Module:
    """Build the DPT-Large architecture via torch.hub (no weights download)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
        try:
            backend = torch.hub.load(
                MODEL_SOURCE, "DPT_Large", pretrained=False, trust_repo=True
            )
        except Exception as exc:
            raise MiDaSError(
                f"Could not build DPT_Large from torch.hub ({MODEL_SOURCE}): {exc}"
            ) from exc
    return backend


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> MiDaSModel:
    """Load the MiDaS DPT-Large checkpoint into a fresh DPT_Large backend.

    Verifies the checkpoint file exists, builds the architecture from hub,
    validates that the checkpoint keys match exactly, and loads them strictly.

    Returns the loaded MiDaSModel wrapper.
    """
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device_obj = resolve_device(device) if not isinstance(device, torch.device) else device

    backend = _build_backend(device_obj)
    _load_weights(backend, checkpoint_path, device_obj)
    model = MiDaSModel(backend=backend, variant="dpt_large").to(device_obj).eval()
    return model


# ------------------------------------------------------------ dataset
_KITTI_INFIX_RE = re.compile(r"_sync_(?:image|groundtruth_depth)_")


def _kitti_stem_key(stem: str) -> str:
    """Normalize KITTI raw stems by collapsing known naming differences.

    KITTI raw image filenames contain ``_sync_image_<frame>_<camera>`` while
    depth filenames contain ``_sync_groundtruth_depth_<frame>_<camera>``. This
    normalizer strips the infix so the same pairing key is produced for both.
    """
    return _KITTI_INFIX_RE.sub("_sync_", stem)


def build_dataset(
    config: dict[str, Any], split: str = DEFAULT_SPLIT
) -> KittiDepthDataset:
    """Build the KITTI dataset for ``split``.

    ``image_size=None`` keeps the native KITTI resolution so that
    predictions can be directly compared with the ground-truth depth maps.
    """
    root = resolve_path(get(config, "data.kitti.root", "data/kitti"))
    if not root.is_dir():
        raise FileNotFoundError(
            f"KITTI dataset not found at {root}. "
            "Evaluation never downloads data; obtain the dataset manually first."
        )
    depth_cap_m = get(config, "eval.depth_cap_m", None)
    return KittiDepthDataset(
        root=root,
        split=split,
        image_size=None,
        scale_mm=1000.0,
        depth_cap_m=depth_cap_m,
        max_samples=None,
        stem_key=_kitti_stem_key,
    )


# ------------------------------------------------------------ evaluation
def evaluate_dataset(
    model: MiDaSModel | torch.nn.Module,
    dataset: KittiDepthDataset,
    device: str | torch.device,
    input_size: int | None = None,
    align: str | None = "median",
    num_workers: int = 0,
    save_predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run ``model`` over ``dataset`` and aggregate per-image depth metrics.

    Each image is independently median-aligned before metric computation.
    Per-image metrics are averaged over the dataset. The predictor handles
    the input transformation and bilinear resize-back to the source
    resolution; predictions are compared against the full-resolution
    ground-truth depth maps.
    """
    predictor = MidDepthPredictor(
        model=model,
        device=device,
        input_size=input_size,
        use_official_transform=False,
    )
    predictor.model.eval()

    save_dir = (
        resolve_path(save_predictions_dir) if save_predictions_dir is not None else None
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = ("rmse", "mae", "abs_rel", "delta1", "delta2", "delta3")
    sums: dict[str, float] = {k: 0.0 for k in metric_keys}
    scales: list[float] = []
    num_samples = 0
    skipped = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_path = Path(sample["image_path"])

        try:
            pil = Image.open(image_path).convert("RGB")
        except Exception as exc:
            LOGGER.warning("Skipping %s: cannot open image (%s)", image_path.name, exc)
            skipped += 1
            continue

        gt = np.asarray(sample["depth"], dtype=np.float64)
        valid = np.asarray(sample["valid_mask"], dtype=bool)

        try:
            pred = predictor.predict(pil).numpy().astype(np.float64)
        except Exception as exc:
            LOGGER.warning("Skipping %s: prediction failed (%s)", image_path.name, exc)
            skipped += 1
            continue

        try:
            m = evaluate_depth(pred, gt, valid, align=align)
        except ValueError as exc:
            LOGGER.warning("Skipping %s: metrics failed (%s)", image_path.name, exc)
            skipped += 1
            continue

        for key in metric_keys:
            sums[key] += m[key]
        if "scale" in m:
            scales.append(m["scale"])

        if save_dir is not None:
            stem = image_path.stem
            np.save(save_dir / f"{stem}_pred.npy", pred.astype(np.float32))

        num_samples += 1

    if num_samples == 0:
        raise RuntimeError(
            "Evaluation produced no samples (empty dataset or all failures?)."
        )

    metrics = {k: sums[k] / num_samples for k in metric_keys}
    metrics["mean_scale"] = float(np.mean(scales)) if scales else None
    metrics["num_samples"] = num_samples
    metrics["skipped"] = skipped
    return metrics


# ------------------------------------------------------------ results JSON
def build_results(
    checkpoint_path: str | Path,
    dataset_len: int,
    metrics: dict[str, Any],
    split: str = DEFAULT_SPLIT,
    align: str | None = "median",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the reproducible JSON result dict."""
    num_samples = int(metrics["num_samples"])
    skipped = int(metrics.get("skipped", 0))
    depth_cap_m = get(config, "eval.depth_cap_m", None) if config is not None else None

    return {
        "model": "MiDaS DPT-Large",
        "checkpoint": str(resolve_path(checkpoint_path)),
        "dataset": "KITTI",
        "split": split,
        "num_samples": num_samples,
        "num_expected": int(dataset_len),
        "skipped": skipped,
        "relative_inverse_depth": True,
        "larger_value_is_closer": True,
        "metric_alignment": align,
        "median_scaling": align == "median",
        "depth_cap_m": depth_cap_m,
        "notes": (
            "Median scaling is evaluation-time alignment only; MiDaS does not "
            "produce metric depth (inverse relative depth). Each image is "
            "independently aligned with scale = median(gt_valid)/"
            "median(pred_valid), and per-image metrics are averaged over "
            "the dataset."
        ),
        "metrics": {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "absrel": float(metrics["abs_rel"]),
            "delta1": float(metrics["delta1"]),
            "delta2": float(metrics["delta2"]),
            "delta3": float(metrics["delta3"]),
        },
        "mean_scale": (
            float(metrics["mean_scale"]) if metrics["mean_scale"] is not None else None
        ),
    }


def write_results(results: dict[str, Any], output_path: str | Path) -> Path:
    """Persist the results dict as readable JSON."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return path


# ------------------------------------------------------------ evaluate checkpoint
def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    output_path: str | Path | None = None,
    save_predictions: bool = False,
    split: str = DEFAULT_SPLIT,
    align: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a saved MiDaS DPT-Large checkpoint on the full KITTI split."""
    if seed is not None:
        set_seed(seed)

    device_obj = resolve_device(device)
    checkpoint_path_resolved = resolve_path(checkpoint_path)

    LOGGER.info("Loading MiDaS model from %s ...", checkpoint_path_resolved)
    model = load_checkpoint(checkpoint_path_resolved, device=device_obj)
    LOGGER.info("MiDaS DPT-Large loaded successfully.")

    dataset = build_dataset(config, split=split)
    if align is None:
        align = get(config, "depth.metric_alignment", "median") or "median"

    LOGGER.info(
        "Evaluating %d KITTI %s samples with MiDaS DPT-Large "
        "(device=%s, align=%s)",
        len(dataset),
        split,
        device_obj,
        align,
    )

    save_dir = (
        resolve_path("outputs/depth/midas_kitti") if save_predictions else None
    )
    input_size = get(config, "preprocessing.input_size", 384)

    metrics = evaluate_dataset(
        model,
        dataset,
        device=device_obj,
        input_size=input_size,
        align=align,
        num_workers=num_workers,
        save_predictions_dir=save_dir,
    )

    results = build_results(
        checkpoint_path_resolved,
        dataset_len=len(dataset),
        metrics=metrics,
        split=split,
        align=align,
        config=config,
    )

    LOGGER.info(
        "Evaluated %d/%d samples (skipped=%d)",
        metrics["num_samples"],
        len(dataset),
        metrics.get("skipped", 0),
    )
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
            align=args.align,
            seed=seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError, MiDaSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path = write_results(results, args.output)

    print(f"Evaluated {results['num_samples']} KITTI {args.split} images")
    print(f"RMSE  : {results['metrics']['rmse']:.6f}")
    print(f"MAE   : {results['metrics']['mae']:.6f}")
    print(f"AbsRel: {results['metrics']['absrel']:.6f}")
    print(f"δ1    : {results['metrics']['delta1']:.6f}")
    print(f"δ2    : {results['metrics']['delta2']:.6f}")
    print(f"δ3    : {results['metrics']['delta3']:.6f}")
    if results["mean_scale"] is not None:
        print(f"Median scale: {results['mean_scale']:.6f}")
    else:
        print("Median scale: N/A")
    print(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
