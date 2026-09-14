"""Step 15 — Full-pipeline evaluation: segmentation + depth + fusion + analysis.

Runs the SAME input RGB image through both U-Net (segmentation) and MiDaS
(relative inverse depth), fuses the outputs, analyzes the scene, computes
difficulty indicators, saves visualizations and structured JSON results.

The same-image contract is enforced: the identical image feeds both models.
Cityscapes and KITTI are NOT paired; this pipeline is standalone.

CLI::

    python -m evaluation.evaluate_pipeline \\
        --config configs/pipeline.yaml \\
        --input-dir data/pipeline/images

    python -m evaluation.evaluate_pipeline \\
        --input-dir data/pipeline/images \\
        --limit 2 --save-visualizations
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

from evaluation.difficulty_analysis import compute_difficulty, DEFAULT_WEIGHTS, DEFAULT_BINS
from scene_understanding.analyzer import analyze_fusion
from scene_understanding.fusion import FusionResult, fuse
from scene_understanding.pipeline import (
    _coerce_image,
    _depth_ndarray,
    _resize_bilinear,
    _resize_nearest_ids,
    _segmentation_ndarray,
)
from utils.config import get, load_config, resolve_path
from utils.device import resolve_device
from utils.logger import get_logger, setup_logging
from utils.seed import set_seed
from visualization.depth import colorize_depth
from visualization.fusion import create_fusion_overlay
from visualization.io import save_figure, save_visualization
from visualization.scene import create_full_visualization, save_scene_report
from visualization.segmentation import colorize_segmentation

LOGGER = get_logger("evaluate_pipeline")

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


# --------------------------------------------------------------------------- #
#  Model loading helpers (reuse existing APIs)
# --------------------------------------------------------------------------- #

def _safe_unet_inference_size(cfg: dict) -> tuple[int, int] | None:
    """Pick a budget-safe U-Net inference resolution from the U-Net config.

    Step 11 (U-Net only) can afford the larger inference resolution, but the
    Step 15/17 pipeline hosts U-Net AND MiDaS DPT-Large on the same device.
    ``configs/unet.yaml`` documents that U-Net at [256, 512] peaks under 1 GiB
    on the 4 GB card, leaving headroom for MiDaS (whose input is bounded to
    ``preprocessing.input_size: 384``). Running the U-Net at the raw source
    resolution would make activation memory grow with arbitrary user images
    (multi-megapixel photos) and exhaust VRAM (CUDA OOM on 4 GB GPUs).

    Returns the smallest configured candidate (``None`` when no size is
    configured). The wrapper still resizes predictions back to the original
    resolution (nearest for class IDs), so final output alignment is preserved.
    """
    candidates = [
        tuple(int(v) for v in size)
        for size in (get(cfg, "inference.image_size"), get(cfg, "data.image_size"))
        if size
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda size: size[0] * size[1])


def _load_unet(
    checkpoint: str | Path,
    cfg: dict,
    device: torch.device,
) -> Any:
    """Load the U-Net inference wrapper with a trained checkpoint.

    Inference resolution is capped at the smallest configured U-Net size so
    the forward pass never runs at the raw (possibly multi-megapixel) input
    resolution. The wrapper resizes predictions back to the source size.
    """
    from models.unet.inference import UNetInference
    from models.unet.model import UNet

    model_cfg = get(cfg, "model", {}) or {}
    num_classes = get(model_cfg, "num_classes", 19)
    model = UNet(num_classes=num_classes)
    return UNetInference(
        model=model,
        device=str(device),
        image_size=_safe_unet_inference_size(cfg),
        checkpoint=str(checkpoint),
    )


def _load_midas(
    weights_path: str | Path,
    cfg: dict,
    device: torch.device,
) -> Any:
    """Load the MiDaS DPT-Large predictor with pretrained weights.

    Uses the same workaround as evaluate_midas.py for the hub name mismatch
    (upstream hub uses ``DPT_Large`` but the project normalises to
    ``dpt_large``).
    """
    import warnings
    from models.midas.inference import MidDepthPredictor
    from models.midas.model import MODEL_SOURCE, MiDaSModel, MiDaSError, _load_weights

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

    _load_weights(backend, weights_path, device)
    midas_model = MiDaSModel(backend=backend, variant="dpt_large").to(device)
    midas_model.eval()
    # Same construction as Step 14 evaluate_midas.evaluate_dataset: the
    # built-in transform pins the DPT-Large input to input_size (384x384),
    # bounding GPU memory regardless of the source image resolution, then
    # resizes the prediction back to the source size.
    input_size = get(cfg, "preprocessing.input_size", 384) or 384
    return MidDepthPredictor(
        model=midas_model,
        device=device,
        input_size=input_size,
        use_official_transform=False,
    )


# --------------------------------------------------------------------------- #
#  Image discovery
# --------------------------------------------------------------------------- #

def collect_images(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Return sorted image paths under *input_dir*, optionally limited."""
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if limit is not None:
        files = files[:limit]
    return files


# --------------------------------------------------------------------------- #
#  Per-image pipeline evaluation
# --------------------------------------------------------------------------- #

def evaluate_single_image(
    image_path: Path,
    seg_predictor: Any,
    depth_predictor: Any,
    cfg: dict,
    *,
    device: torch.device,
) -> dict:
    """Run the full pipeline on one image and return structured results.

    The SAME image object feeds both predictors (same-image contract).
    Returns a JSON-serializable dictionary.
    """
    image = Image.open(image_path).convert("RGB")
    np_image = np.asarray(image, dtype=np.uint8)
    image_size = (int(np_image.shape[0]), int(np_image.shape[1]))

    # ---- predictions (no gradients) ----
    seg_predictor.model.eval()
    depth_predictor.model.eval()

    # Segmentation: single forward pass, optionally returning confidence.
    # Predictors exposing `predict(image, return_confidence=True)` give the
    # class map AND the softmax confidence from the same image in one call.
    mean_confidence: float | None = None
    seg_pred = None
    try:
        seg_out = seg_predictor.predict(np_image, return_confidence=True)
        if isinstance(seg_out, tuple) and len(seg_out) == 2:
            seg_pred, confidence_tensor = seg_out
            conf_np = confidence_tensor.numpy()
            finite_mask = np.isfinite(conf_np)
            if finite_mask.any():
                mean_confidence = float(conf_np[finite_mask].mean())
    except TypeError:
        seg_pred = seg_predictor.predict(np_image)
        mean_confidence = None

    if seg_pred is None:
        seg_pred = seg_predictor.predict(np_image)

    with torch.no_grad():
        depth_pred = depth_predictor.predict(np_image)

    segmentation = _segmentation_ndarray(seg_pred)
    depth = _depth_ndarray(depth_pred)

    # ---- spatial alignment ----
    segmentation = _resize_nearest_ids(segmentation, image_size)
    depth = _resize_bilinear(depth, image_size)

    # ---- fusion ----
    fr = fuse(segmentation, depth)

    # ---- scene analysis ----
    scene_report = analyze_fusion(fr, cfg=cfg)

    # ---- difficulty ----
    difficulty_weights = get(cfg, "difficulty.factor_weights", None)
    difficulty_bins_cfg = get(cfg, "difficulty.bins", None)
    bins = None
    if difficulty_bins_cfg is not None:
        bins = {
            "easy": float(difficulty_bins_cfg.get("easy", DEFAULT_BINS["easy"])),
            "medium": float(difficulty_bins_cfg.get("medium", DEFAULT_BINS["medium"])),
        }
    diff = compute_difficulty(
        fr,
        scene_report=scene_report,
        mean_confidence=mean_confidence,
        weights=difficulty_weights,
        bins=bins,
    )

    # ---- depth statistics ----
    finite_depth = depth[np.isfinite(depth)]
    depth_stats: dict[str, float] = {}
    if finite_depth.size > 0:
        depth_stats = {
            "mean_inverse_depth": round(float(finite_depth.mean()), 6),
            "std_inverse_depth": round(float(finite_depth.std()), 6),
            "min_inverse_depth": round(float(finite_depth.min()), 6),
            "max_inverse_depth": round(float(finite_depth.max()), 6),
        }

    # ---- segmentation statistics ----
    num_classes_present = len(fr.per_class)
    seg_conf_value = round(mean_confidence, 6) if mean_confidence is not None else None
    seg_stats: dict[str, Any] = {
        "num_classes_present": num_classes_present,
        "mean_confidence": seg_conf_value,
    }

    return {
        "image": str(image_path.name),
        "segmentation": seg_stats,
        "depth": depth_stats,
        "scene": scene_report,
        "difficulty": diff.to_dict(),
        "_internals": {
            "image_path": str(image_path),
            "segmentation": segmentation,
            "depth": depth,
            "fusion_result": fr,
        },
    }


# --------------------------------------------------------------------------- #
#  Visualization output
# --------------------------------------------------------------------------- #

def _save_visualizations(
    image_path: Path,
    np_image: np.ndarray,
    segmentation: np.ndarray,
    depth: np.ndarray,
    fr: FusionResult,
    scene_report: dict,
    output_dirs: dict[str, Path],
    cfg: dict,
) -> dict[str, str]:
    """Save per-image visualisation files and return path map."""
    name = image_path.stem
    paths: dict[str, str] = {}

    seg_dir = output_dirs["segmentation"]
    depth_dir = output_dirs["depth"]
    analysis_dir = output_dirs["analysis"]

    paths["original"] = save_visualization(np_image, seg_dir / f"{name}_original.png")
    paths["segmentation"] = save_visualization(
        colorize_segmentation(segmentation),
        seg_dir / f"{name}_segmentation.png",
    )
    paths["depth"] = save_visualization(
        colorize_depth(depth),
        depth_dir / f"{name}_depth.png",
    )

    alpha_seg = float(get(cfg, "visualization.alpha_segmentation", 0.5))
    alpha_depth = float(get(cfg, "visualization.alpha_depth", 0.35))
    paths["fusion"] = save_visualization(
        create_fusion_overlay(np_image, segmentation, depth,
                              alpha_seg=alpha_seg, alpha_depth=alpha_depth),
        analysis_dir / f"{name}_fusion.png",
    )

    figure = create_full_visualization(
        np_image,
        segmentation,
        depth,
        fusion_result=scene_report,
        region_map=fr.region_map,
    )
    paths["overview"] = save_figure(figure, analysis_dir / f"{name}_overview.png")
    paths["scene_report"] = save_scene_report(
        scene_report, analysis_dir / f"{name}_scene_report.json"
    )
    return paths


# --------------------------------------------------------------------------- #
#  Aggregate statistics
# --------------------------------------------------------------------------- #

def _aggregate_results(per_image: list[dict]) -> dict:
    """Compute aggregate statistics across all evaluated images."""
    n = len(per_image)
    if n == 0:
        return {"num_images": 0}

    scores = [r["difficulty"]["score"] for r in per_image]
    levels = [r["difficulty"]["level"] for r in per_image]
    confs = [
        r["segmentation"]["mean_confidence"]
        for r in per_image
        if r["segmentation"]["mean_confidence"] is not None
    ]
    depth_vars = [
        r["depth"]["std_inverse_depth"]
        for r in per_image
        if "std_inverse_depth" in r["depth"]
    ]

    # class occurrence across images
    class_occurrence: dict[str, int] = {}
    for r in per_image:
        for class_name in r["scene"].get("semantic_distribution", {}):
            class_occurrence[class_name] = class_occurrence.get(class_name, 0) + 1

    return {
        "num_images": n,
        "average_difficulty_score": round(float(np.mean(scores)), 6),
        "easy_count": levels.count("easy"),
        "medium_count": levels.count("medium"),
        "hard_count": levels.count("hard"),
        "average_segmentation_confidence": (
            round(float(np.mean(confs)), 6) if confs else None
        ),
        "average_depth_variation": (
            round(float(np.mean(depth_vars)), 6) if depth_vars else None
        ),
        "class_occurrence": class_occurrence,
    }


# --------------------------------------------------------------------------- #
#  Main evaluation loop
# --------------------------------------------------------------------------- #

def run_evaluation(
    input_dir: str | Path,
    *,
    config_name: str = "pipeline",
    unet_checkpoint: str | Path | None = None,
    midas_weights: str | Path | None = None,
    device: str = "auto",
    limit: int | None = None,
    save_visualizations: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """Full evaluation entry point (usable from CLI and tests).

    Returns the complete evaluation result dict.
    """
    cfg = load_config(config_name)
    device_obj = resolve_device(device)

    log_cfg = get(cfg, "logging", {}) or {}
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )
    set_seed(int(get(cfg, "system.seed", 42)))

    LOGGER.info("device: %s", device_obj)
    LOGGER.info("config_hash: %s", __import__("utils.logger", fromlist=["config_hash"]).config_hash(cfg))

    # ---- resolve checkpoint paths ----
    if unet_checkpoint is None:
        unet_checkpoint = resolve_path("checkpoints/unet_cityscapes.pth")
    else:
        unet_checkpoint = Path(unet_checkpoint)
    if midas_weights is None:
        midas_weights = resolve_path("checkpoints/dpt_large_384.pt")
    else:
        midas_weights = Path(midas_weights)

    LOGGER.info("U-Net checkpoint: %s", unet_checkpoint)
    LOGGER.info("MiDaS weights:    %s", midas_weights)

    # ---- load models ----
    LOGGER.info("Loading U-Net ...")
    seg_predictor = _load_unet(unet_checkpoint, load_config("unet"), device_obj)
    LOGGER.info("Loading MiDaS ...")
    depth_predictor = _load_midas(midas_weights, load_config("midas"), device_obj)

    # ---- discover images ----
    input_path = resolve_path(input_dir)
    images = collect_images(input_path, limit=limit)
    if not images:
        LOGGER.warning("No images found in %s", input_path)
        return {"evaluation": {"per_image": [], "aggregate": _aggregate_results([])}}
    LOGGER.info("Found %d image(s) in %s", len(images), input_path)

    # ---- output dirs ----
    if output_dir is None:
        output_base = resolve_path(get(cfg, "output.analysis_dir", "outputs/analysis"))
    else:
        output_base = resolve_path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    seg_out = resolve_path(get(cfg, "output.segmentation_dir", "outputs/segmentation"))
    depth_out = resolve_path(get(cfg, "output.depth_dir", "outputs/depth"))
    analysis_out = output_base
    for d in (seg_out, depth_out, analysis_out):
        d.mkdir(parents=True, exist_ok=True)

    output_dirs = {
        "segmentation": seg_out,
        "depth": depth_out,
        "analysis": analysis_out,
    }

    # ---- evaluate each image ----
    per_image: list[dict] = []
    for idx, img_path in enumerate(images):
        LOGGER.info("[%d/%d] %s", idx + 1, len(images), img_path.name)
        result = evaluate_single_image(
            img_path, seg_predictor, depth_predictor, cfg, device=device_obj,
        )
        per_image.append(result)

        if save_visualizations:
            vis_paths = _save_visualizations(
                img_path,
                np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8),
                result["_internals"]["segmentation"],
                result["_internals"]["depth"],
                result["_internals"]["fusion_result"],
                result["scene"],
                output_dirs,
                cfg,
            )
            result["visualization_paths"] = vis_paths
            for kind, path in vis_paths.items():
                LOGGER.info("  %s: %s", kind, path)

    # ---- strip internals from JSON output ----
    serializable = []
    for r in per_image:
        entry = {k: v for k, v in r.items() if k != "_internals"}
        serializable.append(entry)

    # ---- aggregate ----
    aggregate = _aggregate_results(serializable)

    evaluation = {
        "per_image": serializable,
        "aggregate": aggregate,
    }

    json_path = output_base / "pipeline_evaluation.json"
    json_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved evaluation JSON → %s", json_path)

    return {"evaluation": evaluation, "json_path": str(json_path)}


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_pipeline",
        description=(
            "Full-pipeline evaluation: U-Net segmentation + MiDaS depth → "
            "fusion → scene analysis → difficulty scoring."
        ),
    )
    parser.add_argument(
        "--config", default="pipeline",
        help="Config name under configs/ (default: pipeline).",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing RGB street images.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of images to evaluate.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Compute device: 'auto' (default), 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--unet-checkpoint", default=None,
        help="Path to U-Net checkpoint (default: checkpoints/unet_cityscapes.pth).",
    )
    parser.add_argument(
        "--midas-weights", default=None,
        help="Path to MiDaS weights (default: checkpoints/dpt_large_384.pt).",
    )
    parser.add_argument(
        "--save-visualizations", action="store_true",
        help="Save segmentation / depth / fusion visualisation images.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for pipeline_evaluation.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(
            args.input_dir,
            config_name=args.config,
            unet_checkpoint=args.unet_checkpoint,
            midas_weights=args.midas_weights,
            device=args.device,
            limit=args.limit,
            save_visualizations=args.save_visualizations,
            output_dir=args.output,
        )
        agg = result["evaluation"]["aggregate"]
        print(f"\nImages evaluated : {agg['num_images']}")
        print(f"Avg difficulty   : {agg.get('average_difficulty_score', 'N/A')}")
        print(f"Easy / Med / Hard: {agg.get('easy_count', 0)} / "
              f"{agg.get('medium_count', 0)} / {agg.get('hard_count', 0)}")
        print(f"JSON output      : {result.get('json_path', 'N/A')}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
