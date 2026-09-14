"""Full per-image demo pipeline (orchestration only).

Flow::

    street image
        -> segmentation predictor (e.g. U-Net)  -> [H, W] trainId map
        -> depth predictor (e.g. MiDaS)         -> [H, W] relative inverse depth
        -> fusion                                -> FusionResult
        -> scene analyzer                        -> JSON-serializable report
        -> visualization (optional)              -> files on disk

This module contains NO neural-network logic. Predictors are dependency
injected: any object exposing ``predict(image) -> [H, W] array/tensor`` works
(real wrappers or deterministic dummies). The SAME image object is passed to
both predictors, honouring the same-image contract (never mix datasets).

Depth statements are only ever relative inverse depth (larger = closer);
nothing here produces metric depth or meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scene_understanding.analyzer import analyze_fusion
from scene_understanding.fusion import FusionResult, fuse
from utils.config import get, load_config, resolve_path
from visualization.fusion import create_fusion_overlay
from visualization.scene import (
    create_full_visualization,
    save_scene_report,
)
from visualization.segmentation import colorize_segmentation
from visualization.depth import colorize_depth
from visualization.io import save_figure, save_visualization

DEPTH_CONVENTION = "inverse_relative_larger_closer"


class PipelineError(RuntimeError):
    """Raised when the orchestration layer receives invalid inputs/predictions."""


@dataclass
class PipelineResult:
    """Structured output of :meth:`SceneUnderstandingPipeline.run`.

    Attributes:
        image: ``uint8`` RGB ``[H, W, 3]`` copy of the input image.
        segmentation: Integer ``[H, W]`` trainId class map (aligned to image).
        depth: Float ``[H, W]`` relative inverse depth (aligned to image).
        fusion: :class:`~scene_understanding.fusion.FusionResult`.
        scene_report: JSON-serializable scene report dictionary.
        visualization_paths: Maps output kind -> saved file path (empty when
            visualization is disabled).
    """

    image: np.ndarray
    segmentation: np.ndarray
    depth: np.ndarray
    fusion: FusionResult
    scene_report: dict
    visualization_paths: dict[str, str]

    def to_dict(self) -> dict:
        """Return JSON-serializable metadata (no raw maps / tensors)."""
        h, w = int(self.segmentation.shape[0]), int(self.segmentation.shape[1])
        low, high = self.fusion.thresholds
        return {
            "image_size": [h, w],
            "segmentation_shape": [h, w],
            "depth_shape": [h, w],
            "depth_convention": DEPTH_CONVENTION,
            "fusion_thresholds": {"low": low, "high": high},
            "visualization_paths": dict(self.visualization_paths),
        }


def _coerce_image(image: Any) -> np.ndarray:
    """Return a ``uint8`` HWC RGB array (no copy for uint8 HWC arrays)."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise PipelineError(
                f"image must be an HWC RGB array [H, W, 3], got shape {image.shape}"
            )
        if image.dtype == np.uint8:
            return image
        if image.dtype.kind == "f":
            if image.min() < 0.0 or image.max() > 1.0:
                raise PipelineError(
                    "float images must lie in [0, 1]; use uint8 for 0..255 values"
                )
            return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        return np.clip(image.astype(np.float64), 0, 255).astype(np.uint8)
    raise PipelineError(
        f"unsupported image type {type(image).__name__}; expected a PIL image "
        "or an HWC RGB numpy array"
    )


def _prediction_ndarray(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        array = np.asarray(value.detach().cpu().numpy())
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise PipelineError(
            f"{name} predictor must return a numpy array or torch tensor, "
            f"got {type(value).__name__}"
        )
    if array.ndim != 2:
        raise PipelineError(
            f"{name} prediction must be 2D [H, W], got {array.ndim} dim(s) "
            f"with shape {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise PipelineError(f"{name} prediction has a zero-size dimension: {array.shape}")
    return array


def _segmentation_ndarray(value: Any) -> np.ndarray:
    array = _prediction_ndarray(value, "segmentation")
    if array.dtype.kind not in "iub":
        raise PipelineError(
            f"segmentation prediction must contain integer class IDs, got dtype {array.dtype}"
        )
    return array.astype(np.int64)


def _depth_ndarray(value: Any) -> np.ndarray:
    array = _prediction_ndarray(value, "depth")
    if array.dtype.kind == "b":
        raise PipelineError("depth prediction must not be boolean")
    return array.astype(np.float64)


def _resize_nearest_ids(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(array)
    if tuple(tensor.shape) == size:
        return tensor.numpy()
    four = tensor.unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(four, size=size, mode="nearest-exact")
    return resized.long().squeeze(0).squeeze(0).numpy()


def _resize_bilinear(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(array)
    if tuple(tensor.shape) == size:
        return tensor.numpy()
    four = tensor.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(four, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


class SceneUnderstandingPipeline:
    """Run the full segmentation + depth + fusion + scene analysis on one image.

    Args:
        segmentation_predictor: Object with ``predict(image) -> [H, W]`` integer
            trainId map.
        depth_predictor: Object with ``predict(image) -> [H, W]`` relative
            inverse depth (larger = closer).
        cfg: Pipeline configuration dict (defaults to ``configs/pipeline.yaml``).
        output_dir: Where visualization outputs are written. Defaults to
            ``visualization.output_dir`` from ``cfg``.
        roi_classes / drivable_classes / quantiles: Optional forwarded to the
            fusion/analyzer layers (fall back to config / defaults).
    """

    def __init__(
        self,
        segmentation_predictor: Any,
        depth_predictor: Any,
        cfg: dict | None = None,
        output_dir: str | Path | None = None,
        roi_classes: Sequence[int] | None = None,
        drivable_classes: Sequence[int] | None = None,
        quantiles: Sequence[float] | None = None,
    ) -> None:
        for name, predictor in (
            ("segmentation", segmentation_predictor),
            ("depth", depth_predictor),
        ):
            if not callable(getattr(predictor, "predict", None)):
                raise PipelineError(
                    f"{name} predictor must expose a callable 'predict' method, "
                    f"got {type(predictor).__name__}"
                )
        self.segmentation_predictor = segmentation_predictor
        self.depth_predictor = depth_predictor
        self.cfg = dict(cfg if cfg is not None else load_config("pipeline"))
        self.roi_classes = roi_classes
        self.drivable_classes = drivable_classes
        self.quantiles = quantiles

        if output_dir is None:
            configured = get(self.cfg, "visualization.output_dir", "outputs/visualization")
            output_dir = configured
        self.output_dir = resolve_path(output_dir)

    def _alpha_values(self) -> tuple[float, float]:
        alpha_seg = float(get(self.cfg, "visualization.alpha_segmentation", 0.5))
        alpha_depth = float(get(self.cfg, "visualization.alpha_depth", 0.35))
        return alpha_seg, alpha_depth

    def _write_visualizations(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        depth: np.ndarray,
        fusion: FusionResult,
        scene_report: dict,
        output_dir: Path,
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        alpha_seg, alpha_depth = self._alpha_values()

        paths: dict[str, str] = {}
        paths["segmentation"] = save_visualization(
            colorize_segmentation(segmentation),
            output_dir / "scene_segmentation.png",
        )
        paths["depth"] = save_visualization(
            colorize_depth(depth),
            output_dir / "scene_depth.png",
        )
        paths["fusion"] = save_visualization(
            create_fusion_overlay(
                image,
                segmentation,
                depth,
                alpha_seg=alpha_seg,
                alpha_depth=alpha_depth,
            ),
            output_dir / "scene_fusion.png",
        )
        figure = create_full_visualization(
            image,
            segmentation,
            depth,
            fusion_result=scene_report,
            region_map=fusion.region_map,
        )
        paths["overview"] = save_figure(figure, output_dir / "scene_overview.png")
        paths["scene_report"] = save_scene_report(scene_report, output_dir / "scene_report.json")
        return paths

    def run(
        self,
        image: Any,
        *,
        visualize: bool = True,
        output_dir: str | Path | None = None,
    ) -> PipelineResult:
        """Run the pipeline on a single image and return a :class:`PipelineResult`.

        The exact same image object is passed to both predictors. Predictions
        are aligned to the input resolution using an explicit policy:
        segmentation is resized with nearest-neighbour (class IDs) and depth
        with bilinear interpolation. Raises :class:`PipelineError` on invalid
        inputs or invalid predictions.
        """
        image = _coerce_image(image)
        image_size = (int(image.shape[0]), int(image.shape[1]))

        segmentation = _segmentation_ndarray(self.segmentation_predictor.predict(image))
        depth = _depth_ndarray(self.depth_predictor.predict(image))

        segmentation = _resize_nearest_ids(segmentation, image_size)
        depth = _resize_bilinear(depth, image_size)

        fusion = fuse(
            segmentation,
            depth,
            quantiles=self.quantiles,
        )
        scene_report = analyze_fusion(
            fusion,
            cfg=self.cfg,
            roi_classes=self.roi_classes,
            drivable_classes=self.drivable_classes,
        )

        if visualize:
            out_dir = self.output_dir if output_dir is None else resolve_path(output_dir)
            visualization_paths = self._write_visualizations(
                image, segmentation, depth, fusion, scene_report, out_dir
            )
        else:
            visualization_paths = {}

        return PipelineResult(
            image=image,
            segmentation=segmentation,
            depth=depth,
            fusion=fusion,
            scene_report=scene_report,
            visualization_paths=visualization_paths,
        )