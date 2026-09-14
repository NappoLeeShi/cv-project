"""Offline tests for the visualization layer (no models, no datasets).

Runs fully headless: matplotlib is pinned to the Agg backend up front.
"""

import json
import os
from copy import deepcopy

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from scene_understanding.fusion import (
    REGION_FAR,
    REGION_MIDDLE,
    REGION_NEAR,
    depth_regions,
)
from utils.config import load_config
from visualization import *  # noqa: F401,F403  (re-export smoke test)
from visualization.depth import COLORBAR_LABEL, colorize_depth, create_depth_figure
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

SEG = np.array([[0, 13, 13], [0, 0, 13], [11, 11, 11]], dtype=np.int64)
DEPTH = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
IMAGE = np.tile(np.arange(9, dtype=np.float64).reshape(3, 3)[..., None] / 8.0, (1, 1, 3))


@pytest.fixture
def report():
    return {
        "scene": {
            "height": 3,
            "width": 3,
            "analyzed_pixels": 9,
            "num_semantic_classes": 3,
            "depth_convention": "inverse_relative_larger_closer",
            "depth_thresholds": {"low": 3.66, "high": 6.33},
        },
        "semantic_distribution": {"road": 3, "person": 3, "car": 3},
        "depth_distribution": {"near": 1 / 3, "middle": 1 / 3, "far": 1 / 3},
        "regions": {
            "road": {"pixel_count": 3, "mean_depth": 3.3, "median_depth": 4.0},
            "person": {"pixel_count": 3, "mean_depth": 8.0, "median_depth": 8.0},
            "car": {"pixel_count": 3, "mean_depth": 3.6, "median_depth": 3.0},
        },
        "traffic_context": {
            "vehicles": {"car": {"proximity": "far"}},
            "pedestrians": {"person": {"proximity": "near"}},
            "road": {"road": {"proximity": "middle"}},
            "drivable_coverage_ratio": 1 / 3,
            "drivable_median_depth": 4.0,
            "nearest_dynamic_class": {"class": "person", "proximity": "near"},
        },
        "interpretation": [
            "The nearest dynamic class is 'person' (near relative-depth region).",
        ],
    }


# ---------- segmentation ----------

def test_colorize_segmentation_shape_and_dtype():
    colored = colorize_segmentation(SEG)
    assert colored.shape == (3, 3, 3)
    assert colored.dtype == np.uint8


def test_colorize_segmentation_deterministic():
    assert np.array_equal(colorize_segmentation(SEG), colorize_segmentation(SEG))


def test_colorize_segmentation_palette_is_fixed():
    assert colorize_segmentation(np.array([[0]]))[0, 0].tolist() == list(CITYSCAPES_TRAINID_COLORS[0])
    assert len(CITYSCAPES_TRAINID_COLORS) == 19
    assert all(len(c) == 3 for c in CITYSCAPES_TRAINID_COLORS)


def test_colorize_segmentation_all_19_classes():
    seg = np.arange(19, dtype=np.int64).reshape(1, 19)
    colored = colorize_segmentation(seg)
    for k in range(19):
        assert colored[0, k].tolist() == list(CITYSCAPES_TRAINID_COLORS[k])


def test_colorize_segmentation_ignore_index_is_black():
    seg = np.array([[0, 255, 13]])
    colored = colorize_segmentation(seg)
    assert colored[0, 1].tolist() == [0, 0, 0]
    assert colored[0, 0].tolist() == list(CITYSCAPES_TRAINID_COLORS[0])


def test_colorize_segmentation_custom_palette():
    palette = [(1, 2, 3), (4, 5, 6)]
    seg = np.array([[0, 1, 255]])
    colored = colorize_segmentation(seg, palette=palette)
    assert colored[0, 0].tolist() == [1, 2, 3]
    assert colored[0, 1].tolist() == [4, 5, 6]
    assert colored[0, 2].tolist() == [0, 0, 0]


def test_colorize_segmentation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        colorize_segmentation(np.zeros((3,), dtype=np.int64))
    with pytest.raises(ValueError, match="integer"):
        colorize_segmentation(np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="negative"):
        colorize_segmentation(np.array([[-1]]))
    with pytest.raises(ValueError, match="beyond"):
        colorize_segmentation(np.array([[30]]))
    with pytest.raises(ValueError, match="at least one"):
        colorize_segmentation(SEG, palette=[])


def test_overlay_shape_and_dtype():
    overlay = create_segmentation_overlay(IMAGE, SEG, alpha=0.5)
    assert overlay.shape == (3, 3, 3)
    assert overlay.dtype == np.uint8


def test_overlay_alpha_validation():
    with pytest.raises(ValueError, match="alpha"):
        create_segmentation_overlay(IMAGE, SEG, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        create_segmentation_overlay(IMAGE, SEG, alpha=1.1)


def test_overlay_dose_not_mutate_inputs():
    seg = SEG.copy()
    img = IMAGE.copy()
    create_segmentation_overlay(img, seg, alpha=0.4)
    assert np.array_equal(seg, SEG)
    assert np.array_equal(img, IMAGE)


def test_overlay_full_alpha_equals_colorized():
    overlay = create_segmentation_overlay(IMAGE, SEG, alpha=1.0)
    assert np.array_equal(overlay, colorize_segmentation(SEG))


def test_overlay_shape_mismatch_raises():
    with pytest.raises(ValueError, match="dimensions"):
        create_segmentation_overlay(IMAGE, np.zeros((2, 2), dtype=np.int64), alpha=0.5)


def test_segmentation_legend_returns_figure():
    fig = create_segmentation_legend()
    assert isinstance(fig, Figure)
    axes_titles = {ax.get_title() for ax in fig.axes}
    assert axes_titles == {""}


# ---------- depth ----------

def test_normalization_range():
    from visualization.depth import normalize_depth_for_visualization

    out = normalize_depth_for_visualization(DEPTH)
    assert out.dtype == np.float32
    assert out.shape == DEPTH.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalization_preserves_ordering():
    from visualization.depth import normalize_depth_for_visualization
    out = normalize_depth_for_visualization(DEPTH)
    assert out[0, 0] < out[1, 0] < out[2, 0]
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_normalization_nan_handling():
    from visualization.depth import normalize_depth_for_visualization
    depth = DEPTH.copy()
    depth[1, 1] = np.nan
    out = normalize_depth_for_visualization(depth)
    assert np.isfinite(out).all()
    assert out[1, 1] == 0.0
    all_nan = np.full((3, 3), np.nan)
    assert (normalize_depth_for_visualization(all_nan) == 0.0).all()


def test_normalization_inf_handling():
    from visualization.depth import normalize_depth_for_visualization
    depth = np.array([[1.0, 2.0], [np.inf, -np.inf]])
    out = normalize_depth_for_visualization(depth)
    assert np.isfinite(out).all()
    assert out[1, 1] == 0.0


def test_normalization_constant_map():
    from visualization.depth import normalize_depth_for_visualization
    out = normalize_depth_for_visualization(np.full((3, 3), 4.0))
    assert (out == 0.0).all()


def test_normalization_does_not_mutate_input():
    from visualization.depth import normalize_depth_for_visualization
    depth = DEPTH.copy()
    normalize_depth_for_visualization(depth)
    assert np.array_equal(depth, DEPTH)


def test_colorize_depth_shape_and_dtype():
    colored = colorize_depth(DEPTH)
    assert colored.shape == (3, 3, 3)
    assert colored.dtype == np.uint8


def test_colorize_depth_constant_map_succeeds():
    colored = colorize_depth(np.full((3, 3), 4.0))
    assert colored.shape == (3, 3, 3)


def test_depth_figure_has_relative_label():
    fig = create_depth_figure(DEPTH)
    assert isinstance(fig, Figure)
    labels = {ax.get_ylabel() for ax in fig.axes}
    assert COLORBAR_LABEL in labels
    assert "Relative inverse depth" in " | ".join(labels)


def test_save_figure_works_headless(tmp_path):
    fig = create_depth_figure(DEPTH, title="depth")
    out = tmp_path / "nested" / "dir" / "depth.png"
    save_figure(fig, out)
    assert out.is_file()
    with open(out, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


# ---------- fusion ----------

def test_fusion_overlay_shape_and_dtype():
    fused = create_fusion_overlay(IMAGE, SEG, DEPTH)
    assert fused.shape == (3, 3, 3)
    assert fused.dtype == np.uint8


def test_fusion_overlay_matches_seg_overlay_when_depth_weight_zero():
    fused = create_fusion_overlay(IMAGE, SEG, DEPTH, alpha_seg=0.5, alpha_depth=0.0)
    expected = create_segmentation_overlay(IMAGE, SEG, alpha=0.5)
    assert np.array_equal(fused, expected)


def test_fusion_overlay_depth_information_appears():
    without_depth = create_fusion_overlay(IMAGE, SEG, DEPTH, alpha_depth=0.0)
    with_depth = create_fusion_overlay(IMAGE, SEG, DEPTH, alpha_depth=0.35)
    assert not np.array_equal(with_depth, without_depth)


def test_fusion_overlay_alpha_validation():
    with pytest.raises(ValueError, match="alpha_seg"):
        create_fusion_overlay(IMAGE, SEG, DEPTH, alpha_seg=1.5)
    with pytest.raises(ValueError, match="alpha_depth"):
        create_fusion_overlay(IMAGE, SEG, DEPTH, alpha_depth=-0.2)


def test_fusion_overlay_does_not_mutate_inputs():
    img, seg, dep = IMAGE.copy(), SEG.copy(), DEPTH.copy()
    create_fusion_overlay(img, seg, dep)
    assert np.array_equal(seg, SEG)
    assert np.array_equal(dep, DEPTH)
    assert np.array_equal(img, IMAGE)


def test_region_visualization_shape_and_regions():
    region_map = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int8)
    viz = create_region_visualization(SEG, DEPTH, region_map)
    assert viz.shape == (3, 3, 3)
    assert viz.dtype == np.uint8
    far_row = viz[0].astype(np.float64).mean(axis=0)
    near_row = viz[2].astype(np.float64).mean(axis=0)
    assert not np.allclose(far_row, near_row)
    assert near_row[0] > far_row[0]  # near region is red-dominant


def test_region_visualization_invalid_is_black():
    region_map = np.full((3, 3), -1, dtype=np.int8)
    viz = create_region_visualization(SEG, DEPTH, region_map)
    assert (viz == 0).all()


def test_region_visualization_rejects_unknown_codes():
    region_map = np.full((3, 3), 9, dtype=np.int8)
    with pytest.raises(ValueError, match="unknown code"):
        create_region_visualization(SEG, DEPTH, region_map)


def test_region_visualization_does_not_mutate_inputs():
    region_map = depth_regions(DEPTH, valid=np.ones((3, 3), dtype=bool))[0]
    seg, dep = SEG.copy(), DEPTH.copy()
    create_region_visualization(seg, dep, region_map)
    assert np.array_equal(seg, SEG)
    assert np.array_equal(dep, DEPTH)


# ---------- scene report ----------

def test_scene_summary_minimal_report():
    fig = create_scene_summary({"scene": {"height": 3, "width": 3, "analyzed_pixels": 9}})
    assert isinstance(fig, Figure)


def test_scene_summary_missing_keys_do_not_crash():
    fig = create_scene_summary({})
    assert isinstance(fig, Figure)
    fig2 = create_scene_summary({"scene": {}, "regions": {}, "traffic_context": {}})
    assert isinstance(fig2, Figure)


def test_scene_summary_full_report(report):
    fig = create_scene_summary(report)
    assert isinstance(fig, Figure)


def test_save_scene_report_roundtrip(report, tmp_path):
    out = tmp_path / "scene.json"
    save_scene_report(report, out)
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8")) == report


def test_scene_report_utf8(tmp_path):
    report = {"interpretation": ["Người đi bộ gần"], "scene": {"note": "bầu trời"}}
    out = tmp_path / "scene.json"
    save_scene_report(report, out)
    text = out.read_text(encoding="utf-8")
    assert "Người đi bộ gần" in text
    assert "bầu trời" in text
    assert "\\u" not in text


def test_save_scene_report_does_not_mutate(report, tmp_path):
    snapshot = deepcopy(report)
    save_scene_report(report, tmp_path / "scene.json")
    assert report == snapshot


def test_save_scene_report_validates_extension(tmp_path):
    with pytest.raises(ValueError, match=".json"):
        save_scene_report({}, tmp_path / "scene.txt")


def test_save_scene_report_validates_type(tmp_path):
    with pytest.raises(TypeError, match="dict"):
        save_scene_report([1, 2], tmp_path / "scene.json")


# ---------- save utilities ----------

def test_save_visualization_png_and_parent_dirs(tmp_path):
    out = tmp_path / "a" / "b" / "out.png"
    save_visualization(IMAGE, out)
    assert out.is_file()
    with open(out, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


def test_save_visualization_jpg(tmp_path):
    out = tmp_path / "out.jpg"
    save_visualization(IMAGE, out)
    assert out.is_file()


def test_save_visualization_grayscale_replicated(tmp_path):
    gray = (DEPTH > 5).astype(np.uint8) * 255
    save_visualization(gray, tmp_path / "gray.png")
    assert (tmp_path / "gray.png").is_file()


def test_save_visualization_rejects_bad_extension(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        save_visualization(IMAGE, tmp_path / "out.bmp")


def test_save_figure_rejects_non_figure(tmp_path):
    with pytest.raises(TypeError, match="Figure"):
        save_figure("not a figure", tmp_path / "x.png")


def test_save_figure_jpg(tmp_path):
    fig = create_depth_figure(DEPTH)
    save_figure(fig, tmp_path / "depth.jpg")
    assert (tmp_path / "depth.jpg").is_file()


# ---------- full visualization ----------

@pytest.fixture
def synthetic_scene():
    h, w = 128, 256
    image = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    image[..., 0] = (xx * 255 // w).astype(np.uint8)
    image[..., 1] = (yy * 255 // h).astype(np.uint8)
    image[..., 2] = 128

    seg = np.full((h, w), 10, dtype=np.int64)          # sky
    seg[yy >= h * 2 // 3] = 0                          # road at the bottom
    seg[::16, ::16] = 13                               # scattered cars
    seg[yy < h * 2 // 3][xx[yy < h * 2 // 3] == h // 2] = 255  # void strip
    seg[128 // 2 - 8:128 // 2 + 8, 200:240] = 11       # person block

    depth = (yy.astype(np.float64) / h) * 90.0         # bottom (close) large
    return image, seg, depth


def test_full_visualization_basic(synthetic_scene):
    image, seg, depth = synthetic_scene
    fig = create_full_visualization(image, seg, depth)
    assert isinstance(fig, Figure)
    titles = {ax.get_title() for ax in fig.axes}
    assert "Semantic Segmentation" in titles
    assert "Relative Inverse Depth" in titles
    assert "Segmentation + Relative Depth" in titles


def test_full_visualization_with_regions_and_report(synthetic_scene, report):
    image, seg, depth = synthetic_scene
    region_map, _ = depth_regions(depth, valid=np.ones(depth.shape, dtype=bool))
    fig = create_full_visualization(image, seg, depth, fusion_result=report, region_map=region_map)
    assert isinstance(fig, Figure)
    titles = {ax.get_title() for ax in fig.axes}
    assert "Depth Regions" in titles


def test_full_visualization_saved_headless(synthetic_scene, tmp_path):
    image, seg, depth = synthetic_scene
    fig = create_full_visualization(image, seg, depth)
    out = tmp_path / "overview.png"
    save_figure(fig, out)
    assert out.is_file()


def test_full_visualization_overlay_saved(synthetic_scene, tmp_path):
    image, seg, depth = synthetic_scene
    overlay = create_fusion_overlay(image, seg, depth)
    save_visualization(overlay, tmp_path / "fused.png")
    assert (tmp_path / "fused.png").is_file()


def test_full_visualization_no_mutation(synthetic_scene):
    image, seg, depth = synthetic_scene
    img, s, d = image.copy(), seg.copy(), depth.copy()
    create_full_visualization(img, s, d)
    assert np.array_equal(img, image)
    assert np.array_equal(s, seg)
    assert np.array_equal(d, depth)


# ---------- config ----------

def test_visualization_config_group():
    cfg = load_config("pipeline")
    viz = cfg["visualization"]
    assert viz["enabled"] is True
    assert viz["output_dir"] == "outputs/visualization"
    assert viz["alpha_segmentation"] == 0.5
    assert viz["alpha_depth"] == 0.35
    assert 11 in cfg["fusion"]["roi_classes"]  # existing groups intact