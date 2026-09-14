"""Step 17 — Custom Image Fusion Demo (folder-based inference CLI).

Turns a folder of arbitrary RGB street images (default:
``data/pipeline/images/``) into full scene-understanding reports under
``outputs/custom/``.

For each image the SAME RGB image feeds both models::

    image -> U-Net (segmentation) -> fusion -> scene analysis
         \\-> MiDaS (relative inverse depth) -> difficulty -> visualization/JSON

This module reuses the Step 15 evaluation core (:func:`evaluate_single_image`
from :mod:`evaluation.evaluate_pipeline`) and the shared visualization /
fusion / analyzer / difficulty modules. It contains NO neural-network logic and
NO metric formulas. It is inference-only: no ground truth is read or required
and nothing is retrained.

Depth is always reported as *relative inverse depth* (larger = closer). It is
never described as metric depth in meters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.evaluate_pipeline import (
    SUPPORTED_EXTENSIONS as STEP15_SUPPORTED_EXTENSIONS,
    _load_midas,
    _load_unet,
    evaluate_single_image,
)
from utils.config import get, load_config, resolve_path
from utils.device import resolve_device
from utils.seed import set_seed
from visualization.depth import colorize_depth
from visualization.fusion import create_fusion_overlay
from visualization.io import save_figure, save_visualization
from visualization.scene import create_full_visualization, save_scene_report
from visualization.segmentation import colorize_segmentation

LOGGER = logging.getLogger(__name__)

DEPTH_CONVENTION = "inverse_relative_larger_closer"

# Common RGB formats (Step 15 set plus .webp).
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(
    dict.fromkeys((*STEP15_SUPPORTED_EXTENSIONS, ".webp"))
)

DEFAULT_UNET_CHECKPOINT = "checkpoints/unet_cityscapes.pth"
DEFAULT_MIDAS_WEIGHTS = "checkpoints/dpt_large_384.pt"
DEFAULT_OUTPUT_DIR = "outputs/custom"

_GRAYSCALE_MODES = {"L", "I", "I;16", "F"}


class CustomDemoError(RuntimeError):
    """Raised for clear, user-facing failures of the custom-image demo."""


# --------------------------------------------------------------------------- #
#  Input discovery and validation
# --------------------------------------------------------------------------- #

def validate_checkpoints(
    unet_checkpoint: str | Path, midas_weights: str | Path
) -> tuple[Path, Path]:
    """Resolve and verify both checkpoints exist; return their resolved paths.

    Raises a clear :class:`CustomDemoError` naming the missing file.
    """
    unet_path = resolve_path(unet_checkpoint)
    midas_path = resolve_path(midas_weights)
    if not unet_path.is_file():
        raise CustomDemoError(f"U-Net checkpoint file not found: {unet_path}")
    if not midas_path.is_file():
        raise CustomDemoError(f"MiDaS weights file not found: {midas_path}")
    return unet_path, midas_path


def discover_images(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Return the sorted supported-image paths under *input_dir*.

    Unsupported files are ignored (never processed). ``limit`` truncates the
    list (useful for quick tests/demos).
    """
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if limit is not None:
        files = files[:limit]
    return files


def load_rgb_image(path: Path) -> tuple[np.ndarray, bool]:
    """Load and validate a single RGB street image.

    Corrupted files and invalid shapes raise a clear :class:`CustomDemoError`
    that names the offending file. Grayscale images are converted to RGB with a
    warning (the project convention converts to RGB everywhere else).

    Returns ``(uint8 [H, W, 3] array, is_grayscale_source)``.
    """
    try:
        with Image.open(path) as pil:
            pil.load()
            mode = pil.mode
            rgb = pil.convert("RGB")
    except Exception as exc:  # UnidentifiedImageError, OSError, ...
        raise CustomDemoError(f"cannot read image file {path}: {exc}") from exc

    if mode in _GRAYSCALE_MODES:
        LOGGER.warning("converting grayscale image %s to RGB", path.name)

    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise CustomDemoError(f"invalid image shape for {path}: {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise CustomDemoError(f"invalid zero-sized image: {path}")

    return array, mode in _GRAYSCALE_MODES


# --------------------------------------------------------------------------- #
#  Per-image report assembly and outputs
# --------------------------------------------------------------------------- #

def build_scene_report(
    image_path: Path,
    result: dict[str, Any],
    fusion: Any,
) -> dict:
    """Assemble the per-image scene report JSON (existing formats reused).

    Combines image information, segmentation statistics, depth statistics
    (relative inverse depth), fusion information, the scene analysis report
    (existing analyzer output) and the difficulty score/level/components
    (existing difficulty output format).
    """
    internals = result["_internals"]
    segmentation = internals["segmentation"]
    return {
        "image": {
            "filename": image_path.name,
            "path": str(image_path),
            "height": int(segmentation.shape[0]),
            "width": int(segmentation.shape[1]),
            "mode": "RGB",
        },
        "segmentation": dict(result["segmentation"]),
        "depth": {**dict(result["depth"]), "convention": DEPTH_CONVENTION},
        "fusion": {
            "analyzed_pixels": int(fusion.analyzed_mask.sum()),
            "depth_regions": {
                name: dict(fusion.region_summary[name])
                for name in ("far", "middle", "near")
            },
            "thresholds": {"low": fusion.thresholds[0], "high": fusion.thresholds[1]},
            "depth_convention": DEPTH_CONVENTION,
        },
        "scene": dict(result["scene"]),
        "difficulty": dict(result["difficulty"]),
    }


def save_custom_visualizations(
    image_path: Path,
    np_image: np.ndarray,
    segmentation: np.ndarray,
    depth: np.ndarray,
    fusion: Any,
    scene_report: dict,
    custom_report: dict,
    output_dir: Path,
    cfg: dict,
) -> dict[str, Path]:
    """Write per-image outputs into *output_dir* using the ``<stem>_<kind>``
    naming scheme of Step 17. Reuses the existing visualization modules only.
    """
    stem = image_path.stem
    paths: dict[str, Path] = {}

    paths["original"] = Path(
        save_visualization(np_image, output_dir / f"{stem}_original.png")
    )
    paths["segmentation"] = Path(
        save_visualization(
            colorize_segmentation(segmentation),
            output_dir / f"{stem}_segmentation.png",
        )
    )
    paths["depth"] = Path(
        save_visualization(colorize_depth(depth), output_dir / f"{stem}_depth.png")
    )

    alpha_seg = float(get(cfg, "visualization.alpha_segmentation", 0.5))
    alpha_depth = float(get(cfg, "visualization.alpha_depth", 0.35))
    paths["fusion"] = Path(
        save_visualization(
            create_fusion_overlay(
                np_image,
                segmentation,
                depth,
                alpha_seg=alpha_seg,
                alpha_depth=alpha_depth,
            ),
            output_dir / f"{stem}_fusion.png",
        )
    )

    figure = create_full_visualization(
        np_image,
        segmentation,
        depth,
        fusion_result=scene_report,
        region_map=fusion.region_map,
    )
    paths["overview"] = Path(save_figure(figure, output_dir / f"{stem}_overview.png"))
    paths["scene_report"] = Path(
        save_scene_report(custom_report, output_dir / f"{stem}_scene_report.json")
    )
    return paths


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #

def build_real_predictors(
    unet_checkpoint: str | Path,
    midas_weights: str | Path,
    device: str,
):
    """Build real U-Net + MiDaS predictors from the project checkpoints.

    Checkpoints must exist (clear error otherwise); model loading reuses the
    Step 15 helpers. Never downloads weights.
    """
    unet_path, midas_path = validate_checkpoints(unet_checkpoint, midas_weights)
    device_obj = resolve_device(device)
    seg_predictor = _load_unet(unet_path, load_config("unet"), device_obj)
    depth_predictor = _load_midas(midas_path, load_config("midas"), device_obj)
    return seg_predictor, depth_predictor


def run_custom_demo(
    input_dir: str | Path,
    *,
    unet_checkpoint: str | Path = DEFAULT_UNET_CHECKPOINT,
    midas_weights: str | Path = DEFAULT_MIDAS_WEIGHTS,
    device: str = "auto",
    limit: int | None = None,
    output_dir: str | Path | None = None,
    save_visualizations: bool = True,
    seg_predictor: Any = None,
    depth_predictor: Any = None,
    config_name: str = "pipeline",
) -> dict[str, Any]:
    """Run the folder-based custom-image demo.

    Discovers supported images under *input_dir*, processes each (same RGB
    image through both predictors -> alignment -> fusion -> scene analysis ->
    difficulty -> visualization -> JSON report) and writes per-image outputs to
    *output_dir* (default ``outputs/custom/``).

    ``seg_predictor`` / ``depth_predictor`` are optional for tests: provide
    mocked/synthetic predictors to avoid loading real checkpoints.

    Returns a JSON-serializable summary.
    """
    cfg = load_config(config_name)
    device_obj = resolve_device(device)
    set_seed(int(get(cfg, "system.seed", 42)))

    input_path = resolve_path(input_dir)
    if not input_path.is_dir():
        raise CustomDemoError(
            f"input directory not found: {input_path} "
            "(put street images into data/pipeline/images/)."
        )

    images = discover_images(input_path, limit=limit)
    if not images:
        raise CustomDemoError(
            f"no supported images found in {input_path} "
            f"(supported extensions: {', '.join(SUPPORTED_EXTENSIONS)})."
        )
    LOGGER.info("discovered %d supported image(s) in %s", len(images), input_path)

    custom_out = resolve_path(output_dir or DEFAULT_OUTPUT_DIR)
    custom_out.mkdir(parents=True, exist_ok=True)

    if seg_predictor is None or depth_predictor is None:
        seg_predictor, depth_predictor = build_real_predictors(
            unet_checkpoint, midas_weights, device=str(device_obj)
        )

    summary_images: list[dict[str, Any]] = []
    for idx, image_path in enumerate(images):
        LOGGER.info("[%d/%d] processing %s", idx + 1, len(images), image_path.name)
        np_image, _ = load_rgb_image(image_path)

        try:
            result = evaluate_single_image(
                image_path, seg_predictor, depth_predictor, cfg, device=device_obj
            )
        except CustomDemoError:
            raise
        except Exception as exc:
            raise CustomDemoError(
                f"failed to process {image_path.name}: {exc}"
            ) from exc

        internals = result["_internals"]
        fusion = internals["fusion_result"]
        scene_report = result["scene"]
        custom_report = build_scene_report(image_path, result, fusion)

        if save_visualizations:
            paths = save_custom_visualizations(
                image_path,
                np_image,
                internals["segmentation"],
                internals["depth"],
                fusion,
                scene_report,
                custom_report,
                custom_out,
                cfg,
            )
        else:
            paths = {
                "scene_report": Path(
                    save_scene_report(
                        custom_report, custom_out / f"{image_path.stem}_scene_report.json"
                    )
                )
            }

        difficulty = result["difficulty"]
        summary_images.append(
            {
                "filename": image_path.name,
                "output_dir": str(custom_out),
                "visualization_paths": {
                    kind: str(path) for kind, path in paths.items()
                },
                "difficulty_score": difficulty["score"],
                "difficulty_level": difficulty["level"],
            }
        )
        for kind, path in paths.items():
            LOGGER.info("  %s: %s", kind, path)

    return {
        "input_dir": str(input_path),
        "output_dir": str(custom_out),
        "num_images": len(images),
        "images": summary_images,
    }