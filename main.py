#!/usr/bin/env python3
"""Demo entry point for the full segmentation + depth pipeline.

Usage::

    python main.py --image path/to/image.jpg \\
        --unet-checkpoint path/to/unet.pt \\   # required for real inference
        --midas-weights path/to/midas.pt \\    # required for real inference
        [--output-dir OUT] [--no-visualization] [--device auto|cpu|cuda]

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
            "(U-Net + MiDaS -> fusion -> scene understanding -> visualization)."
        ),
    )
    parser.add_argument(
        "--image", "-i", required=True,
        help="Path to a single RGB street image (PNG/JPG).",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Where visualization outputs are written "
             "(default: configs/pipeline.yaml > visualization.output_dir).",
    )
    parser.add_argument(
        "--unet-checkpoint", default=None,
        help="Path to the U-Net checkpoint (required for real inference).",
    )
    parser.add_argument(
        "--midas-weights", default=None,
        help="Path to local MiDaS weights (required for real inference).",
    )
    parser.add_argument(
        "--no-visualization", action="store_true",
        help="Skip writing visualization outputs.",
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
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
    except (PipelineError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())