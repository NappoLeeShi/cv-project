"""Visualization layer: read-only rendering of pipeline outputs.

Segmentation, relative inverse depth, fusion and scene-understanding results
are turned into RGB images / matplotlib figures and saved to disk only on
explicit request. No module here modifies model predictions, depth
conventions, or the fusion/analyzer logic.
"""

from __future__ import annotations

from visualization.depth import (
    COLORBAR_LABEL,
    DEFAULT_COLORMAP,
    colorize_depth,
    create_depth_figure,
    normalize_depth_for_visualization,
)
from visualization.fusion import (
    REGION_COLORS,
    create_fusion_overlay,
    create_region_visualization,
)
from visualization.io import save_figure, save_visualization
from visualization.scene import (
    create_full_visualization,
    create_scene_summary,
    save_scene_report,
)
from visualization.segmentation import (
    CITYSCAPES_TRAINID_COLORS,
    colorize_segmentation,
    create_segmentation_legend,
    create_segmentation_overlay,
)

__all__ = [
    "CITYSCAPES_TRAINID_COLORS",
    "COLORBAR_LABEL",
    "DEFAULT_COLORMAP",
    "REGION_COLORS",
    "colorize_depth",
    "colorize_segmentation",
    "create_depth_figure",
    "create_full_visualization",
    "create_fusion_overlay",
    "create_region_visualization",
    "create_scene_summary",
    "create_segmentation_legend",
    "create_segmentation_overlay",
    "normalize_depth_for_visualization",
    "save_figure",
    "save_scene_report",
    "save_visualization",
]