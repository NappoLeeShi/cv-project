"""Scene-understanding and combined visualization.

Consumes the JSON-serializable scene report produced by
``scene_understanding.analyzer`` as read-only input (missing keys are skipped,
never crashed on) and lays out figure montages for demo/reports.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from scene_understanding.fusion import DEPTH_REGION_NAMES
from visualization.depth import normalize_depth_for_visualization
from visualization.fusion import create_fusion_overlay, create_region_visualization
from visualization.segmentation import (
    _validate_rgb_image,
    _validate_segmentation,
    colorize_segmentation,
)


def _wrap(text: str, width: int = 96) -> str:
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        while len(paragraph) > width:
            cut = paragraph.rfind(" ", 0, width)
            if cut < 1:
                cut = width
            lines.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        lines.append(paragraph)
    return "\n".join(lines)


def _text_panel(fig: Figure, grid, title: str, body: str) -> None:
    ax = fig.add_subplot(grid)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    wrapped = _wrap(body)
    ax.text(0.0, 1.0, wrapped, va="top", ha="left", fontsize=9,
            transform=ax.transAxes, linespacing=1.4)
    if not body.strip():
        ax.text(0.0, 1.0, "no data available", va="top", ha="left",
                fontsize=9, color="0.4", transform=ax.transAxes)


def _scene_data_panel(fig: Figure, grid, report: dict) -> None:
    ax = fig.add_subplot(grid)
    ax.set_title("Scene", fontsize=11, loc="left", pad=8)
    ax.axis("off")
    scene = report.get("scene")
    if not isinstance(scene, dict):
        ax.text(0.0, 1.0, "no data available", va="top", ha="left",
                transform=ax.transAxes, color="0.4")
        return
    lines = []
    for key in (
        "height",
        "width",
        "analyzed_pixels",
        "num_semantic_classes",
        "object_class_pixel_ratio",
        "depth_convention",
    ):
        value = scene.get(key)
        if value is not None:
            lines.append(f"{key}: {value}")
    thresholds = scene.get("depth_thresholds")
    if isinstance(thresholds, dict) and "low" in thresholds and "high" in thresholds:
        lines.append(f"region (low, high): {thresholds['low']} .. {thresholds['high']}")
    ax.text(0.0, 1.0, _wrap("\n".join(lines)), va="top", ha="left",
            transform=ax.transAxes, fontsize=9, linespacing=1.4)


def create_scene_summary(report: dict) -> Figure:
    """Build a summary Figure from a scene report dictionary.

    Optional/missing report keys are skipped instead of crashing. Never
    modifies ``report``.
    """
    fig = Figure(figsize=(13.0, 8.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8],
                            hspace=0.45, wspace=0.25)

    semantic = report.get("semantic_distribution")
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("Semantic Distribution", fontsize=11, pad=8)
    if isinstance(semantic, dict) and semantic:
        names = sorted(semantic, key=lambda k: semantic[k])
        counts = [semantic[n] for n in names]
        ax.barh(names, counts)
        ax.set_xlabel("pixel count")
    else:
        ax.text(0.5, 0.5, "no data available", ha="center", va="center",
                color="0.4", transform=ax.transAxes)

    depth = report.get("depth_distribution")
    ax = fig.add_subplot(grid[0, 1])
    ax.set_title("Depth Distribution (relative inverse)", fontsize=11, pad=8)
    if isinstance(depth, dict) and depth:
        order = [name for name in DEPTH_REGION_NAMES if name in depth]
        values = [depth[name] for name in order]
        ax.bar(order, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("pixel ratio")
    else:
        ax.text(0.5, 0.5, "no data available", ha="center", va="center",
                color="0.4", transform=ax.transAxes)

    regions = report.get("regions")
    body = ""
    if isinstance(regions, dict) and regions:
        for name, stats in regions.items():
            if not isinstance(stats, dict):
                continue
            count = stats.get("pixel_count")
            mean = stats.get("mean_depth")
            median = stats.get("median_depth")
            parts = []
            if count is not None:
                parts.append(f"{count}px")
            if mean is not None:
                parts.append(f"mean {mean:.3f}")
            if median is not None:
                parts.append(f"median {median:.3f}")
            body += f"{name}: {', '.join(parts)}\n"
    _text_panel(fig, grid[1, 0], "Region Summary", body)

    traffic = report.get("traffic_context")
    body = ""
    if isinstance(traffic, dict):
        nearest = traffic.get("nearest_dynamic_class")
        if isinstance(nearest, dict):
            body += f"nearest dynamic class: {nearest.get('class')} ({nearest.get('proximity')})\n"
        vehicles = {k: (v or {}).get("proximity") for k, v in traffic.get("vehicles", {}).items()}
        if vehicles:
            body += "vehicles: " + ", ".join(f"{k}={v}" for k, v in vehicles.items()) + "\n"
        pedestrians = traffic.get("pedestrians", {})
        if pedestrians:
            body += "pedestrians: " + ", ".join(
                f"{k}={v.get('proximity')}" for k, v in pedestrians.items()) + "\n"
        road = traffic.get("road", {})
        if road:
            body += "road: " + ", ".join(f"{k}={v.get('proximity')}" for k, v in road.items()) + "\n"
        if traffic.get("drivable_coverage_ratio") is not None:
            body += f"drivable coverage: {traffic['drivable_coverage_ratio']:.3f}\n"
        if traffic.get("drivable_median_depth") is not None:
            body += f"drivable median depth: {traffic['drivable_median_depth']:.3f}\n"
    _text_panel(fig, grid[1, 1], "Traffic Context", body)

    interpretation = report.get("interpretation")
    body = ""
    if isinstance(interpretation, list):
        body = "\n".join(str(line) for line in interpretation)
    _text_panel(fig, grid[2, :], "Interpretation", body)

    _scene_data_panel(fig, grid[0, :], report)
    fig.suptitle("Scene Understanding Summary", fontsize=14)
    return fig


def save_scene_report(report: dict, output_path) -> str:
    """Save a scene report dictionary as UTF-8 JSON (indent=2).

    Raises ``TypeError`` when the report is not JSON-serializable. The report
    dictionary itself is never modified.
    """
    if not isinstance(report, dict):
        raise TypeError(f"scene report must be a dict, got {type(report).__name__}")
    path = Path(os.fspath(output_path))
    if path.suffix.lower() != ".json":
        raise ValueError("scene report must be saved with a .json extension")
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return str(path)


def create_full_visualization(
    image: np.ndarray,
    segmentation: np.ndarray,
    depth: np.ndarray,
    fusion_result: dict | None = None,
    region_map: np.ndarray | None = None,
) -> Figure:
    """Build a single overview Figure for demo/report purposes.

    Panels (minimum 4): original image, semantic segmentation, relative
    inverse depth, and the fusion overlay. A region map and/or a scene report
    add extra panels when provided. Titles are fixed and depth is never
    labelled as meters. No inputs are modified.
    """
    img = _validate_rgb_image(image)
    seg = _validate_segmentation(segmentation)
    depth_arr = np.asarray(depth)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth map must be 2D [H, W], got {depth_arr.ndim} dim(s)")
    if depth_arr.shape != img.shape[:2]:
        raise ValueError(
            "image and depth must share spatial dimensions, "
            f"got {img.shape[:2]} vs {depth_arr.shape}"
        )
    if depth_arr.shape != seg.shape:
        raise ValueError(
            "segmentation and depth must share spatial dimensions, "
            f"got {seg.shape} vs {depth_arr.shape}"
        )

    panels: list[tuple[str, np.ndarray, bool]] = [
        ("Input Image", img if img.dtype == np.uint8
         else np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8), False),
        ("Semantic Segmentation", colorize_segmentation(seg), False),
        ("Relative Inverse Depth", normalize_depth_for_visualization(depth_arr), True),
        ("Segmentation + Relative Depth", create_fusion_overlay(img, seg, depth_arr), False),
    ]
    if region_map is not None:
        panels.append(("Depth Regions", create_region_visualization(seg, depth_arr, region_map), False))
    if fusion_result is not None:
        panels.append(("Scene Report", fusion_result, False))

    ncols = 4
    nrows = int(math.ceil(len(panels) / ncols))
    fig = Figure(figsize=(4.4 * min(ncols, len(panels)), 3.4 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    flat = [ax for row in axes for ax in row]

    for i, (title, data, is_depth) in enumerate(panels):
        ax = flat[i]
        if is_depth:
            im = ax.imshow(data, cmap="turbo", norm=Normalize(vmin=0.0, vmax=1.0))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                         label="relative inverse depth (larger = closer)")
        elif isinstance(data, dict):
            create_scene_summary_text_into(ax, data)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle("Segmentation + Depth Fusion Overview", fontsize=14)
    return fig


def create_scene_summary_text_into(ax, report: dict) -> None:
    """Render a compact text summary of a scene report inside one axis."""
    ax.set_axis_off()
    lines = []
    interpretation = report.get("interpretation")
    if isinstance(interpretation, list):
        lines.extend(str(line) for line in interpretation)
    traffic = report.get("traffic_context")
    if isinstance(traffic, dict):
        nearest = traffic.get("nearest_dynamic_class")
        if isinstance(nearest, dict):
            lines.append(f"nearest dynamic class: {nearest.get('class')} ({nearest.get('proximity')})")
    depth = report.get("depth_distribution")
    if isinstance(depth, dict) and "near" in depth:
        lines.append(f"near-region fraction: {depth['near']:.3f}")
    ax.text(0.02, 0.98, _wrap("\n".join(lines)) or "no data available",
            va="top", ha="left", fontsize=9, linespacing=1.5,
            transform=ax.transAxes)