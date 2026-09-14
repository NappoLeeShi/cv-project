#!/usr/bin/env python3
"""Demo entry point for the full segmentation + depth pipeline.

Single-image usage (existing behavior)::

    python main.py --image path/to/image.jpg \\
        --unet-checkpoint path/to/unet.pt \\   # required for real inference
        --midas-weights path/to/midas.pt \\    # required for real inference
        [--output-dir OUT] [--no-visualization] [--device auto|cpu|cuda]

Custom-image folder demo (Step 17)::

    python main.py --input-dir data/pipeline/images        # all images
    python main.py --input-dir data/pipeline/images --limit 1   # quick test
    python main.py --input-dir data/pipeline/images --device auto

Files in ``data/pipeline/images/`` are discovered automatically (no individual
image path required) and per-image results are written to ``outputs/custom/``
using the project checkpoints ``checkpoints/unet_cityscapes.pth`` and
``checkpoints/dpt_large_384.pt``.

This script never downloads U-Net weights, MiDaS weights, Cityscapes or KITTI.
Real inference fails fast with a clear error when either model input is
missing. Offline checks use the dummy predictors in ``tests/test_pipeline.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scene_understanding.pipeline import PipelineError, SceneUnderstandingPipeline
from utils.config import load_config, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run the full segmentation + depth pipeline on one street image "
            "(U-Net + MiDaS -> fusion -> scene understanding -> visualization) "
            "or on a whole folder of images (custom-image demo)."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image", "-i", default=None,
        help="Path to a single RGB street image (PNG/JPG).",
    )
    source.add_argument(
        "--input-dir", default=None,
        help="Directory of RGB street images to process (custom-image demo).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of images to process in --input-dir mode "
             "(only during testing/demo).",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Where visualization outputs are written (default: "
             "configs/pipeline.yaml > visualization.output_dir for --image, "
             "outputs/custom/ for --input-dir).",
    )
    parser.add_argument(
        "--unet-checkpoint", default=None,
        help="Path to the U-Net checkpoint. Required for --image mode; "
             "defaults to checkpoints/unet_cityscapes.pth in --input-dir mode.",
    )
    parser.add_argument(
        "--midas-weights", default=None,
        help="Path to local MiDaS weights. Required for --image mode; "
             "defaults to checkpoints/dpt_large_384.pt in --input-dir mode.",
    )
    parser.add_argument(
        "--no-visualization", action="store_true",
        help="Skip writing visualization images (scene-report JSON is still saved).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Compute device: 'auto' (default), 'cpu' or 'cuda'.",
    )
    return parser


def require_image_file(image: str) -> Path:
    """Check the CLI image exists; raise a clear ``ValueError`` otherwise."""
    path = Path(image).expanduser()
    if not path.is_file():
        raise ValueError(f"image file not found: {path}")
    return path


def require_predictor_inputs(unet_checkpoint: str | None, midas_weights: str | None) -> None:
    """Fail fast when a real-demo model input is missing/invalid.

    The demo never downloads weights; missing inputs are a hard error.
    """
    if not unet_checkpoint:
        raise RuntimeError(
            "U-Net checkpoint is required for real inference; pass --unet-checkpoint PATH"
        )
    if not Path(unet_checkpoint).expanduser().is_file():
        raise RuntimeError(f"U-Net checkpoint file not found: {unet_checkpoint}")

    if not midas_weights:
        raise RuntimeError(
            "MiDaS weights are required for real inference; pass --midas-weights PATH"
        )
    if not Path(midas_weights).expanduser().is_file():
        raise RuntimeError(f"MiDaS weights file not found: {midas_weights}")


def build_real_predictors(args: argparse.Namespace):
    """Build U-Net + MiDaS real predictors from the CLI arguments.

    Lazily imports the model wrappers so that parsing/tests never trigger
    model loading. Raises the underlying clear errors when checkpoints fail.
    """
    from models.midas.inference import MidDepthPredictor
    from models.midas.model import build_midas_model
    from models.unet.inference import UNetInference

    unet_cfg = load_config("unet", overrides={"inference": {"device": args.device}})
    unet = UNetInference.from_config(unet_cfg, checkpoint=args.unet_checkpoint)

    midas_cfg = load_config("midas", overrides={"inference": {"device": args.device}})
    midas_model = build_midas_model(
        variant=midas_cfg["model"]["variant"],
        weights_path=args.midas_weights,
        device=args.device,
    )
    midas = MidDepthPredictor.from_config(midas_cfg, model=midas_model)
    return unet, midas


def _run_batch_demo(args: argparse.Namespace) -> int:
    """Run the Step 17 folder-based custom-image demo."""
    from evaluation.custom_demo import (
        DEFAULT_MIDAS_WEIGHTS,
        DEFAULT_UNET_CHECKPOINT,
        run_custom_demo,
    )

    output_dir = args.output_dir or None
    summary = run_custom_demo(
        args.input_dir,
        unet_checkpoint=args.unet_checkpoint or DEFAULT_UNET_CHECKPOINT,
        midas_weights=args.midas_weights or DEFAULT_MIDAS_WEIGHTS,
        device=args.device,
        limit=args.limit,
        output_dir=output_dir,
        save_visualizations=not args.no_visualization,
    )

    print(f"input dir        : {summary['input_dir']}")
    print(f"images processed : {summary['num_images']}")
    for entry in summary["images"]:
        print(f"  - {entry['filename']:<30} difficulty={entry['difficulty_score']}"
              f" ({entry['difficulty_level']})")
    print(f"output dir       : {summary['output_dir']}")
    return 0


def _run_single_image(args: argparse.Namespace) -> int:
    """Run the existing single-image pipeline (backward-compatible)."""
    require_predictor_inputs(args.unet_checkpoint, args.midas_weights)
    image_path = require_image_file(args.image)

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = resolve_path(
            load_config("pipeline")["visualization"]["output_dir"]
        )

    unet, midas = build_real_predictors(args)
    pipeline = SceneUnderstandingPipeline(unet, midas, output_dir=output_dir)
    result = pipeline.run(image_path, visualize=not args.no_visualization)

    print(f"image            : {image_path}")
    print(f"segmentation     : {result.segmentation.shape} trainId map")
    print(f"depth            : {result.depth.shape} relative inverse depth")
    print(f"nearest dynamic  : "
          f"{result.scene_report['traffic_context']['nearest_dynamic_class']}")
    if result.visualization_paths:
        print("outputs          :")
        for kind, path in result.visualization_paths.items():
            print(f"  - {kind:<12}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.input_dir is not None:
            return _run_batch_demo(args)
        return _run_single_image(args)
    except (PipelineError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())