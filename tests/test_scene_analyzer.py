import json

import numpy as np
import pytest
import torch

from scene_understanding.analyzer import analyze_maps, analyze_scene
from scene_understanding.fusion import fuse

SEG = np.array([[0, 13, 13], [0, 0, 13], [11, 11, 11]], dtype=np.int64)
DEPTH = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)


@pytest.fixture
def report():
    return analyze_maps(SEG, DEPTH)


class _StubSegPredictor:
    def __init__(self, mask):
        self.mask = mask
        self.last_input = None

    def predict(self, image):
        self.last_input = image
        return torch.from_numpy(self.mask)


class _StubDepthPredictor:
    def __init__(self, depth):
        self.depth = depth
        self.last_input = None

    def predict(self, image):
        self.last_input = image
        return torch.from_numpy(self.depth)


# 11. road analysis
def test_road_analysis(report):
    road = report["traffic_context"]["road"]["road"]
    assert road["pixel_count"] == 3
    assert road["median_depth"] == 4.0
    assert road["proximity"] == "middle"


# 12. car analysis
def test_car_analysis(report):
    car = report["traffic_context"]["vehicles"]["car"]
    assert car["pixel_count"] == 3
    assert car["median_depth"] == 3.0
    assert car["proximity"] == "far"


# 13. person analysis
def test_person_analysis(report):
    person = report["traffic_context"]["pedestrians"]["person"]
    assert person["median_depth"] == 8.0
    assert person["proximity"] == "near"


# 14. multiple classes
def test_region_blocks_for_present_classes(report):
    assert set(report["regions"]) == {"road", "car", "person"}
    assert "sidewalk" not in report["regions"]


def test_semantic_distribution(report):
    assert report["semantic_distribution"] == {"road": 3, "person": 3, "car": 3}


# 15. scene-level statistics
def test_scene_level_statistics(report):
    scene = report["scene"]
    assert scene["height"] == 3
    assert scene["width"] == 3
    assert scene["analyzed_pixels"] == 9
    assert scene["num_semantic_classes"] == 3
    assert scene["object_class_pixel_ratio"] == pytest.approx(6 / 9)


# 16. structured output
def test_structured_output_top_level_keys(report):
    assert set(report) == {
        "scene",
        "semantic_distribution",
        "depth_distribution",
        "regions",
        "traffic_context",
        "interpretation",
    }


def test_output_is_json_serializable(report):
    payload = json.dumps(report)
    loaded = json.loads(payload)
    assert loaded["scene"]["analyzed_pixels"] == 9


def test_depth_distribution(report):
    assert report["depth_distribution"]["near"] == pytest.approx(1 / 3)
    assert report["depth_distribution"]["middle"] == pytest.approx(1 / 3)
    assert report["depth_distribution"]["far"] == pytest.approx(1 / 3)


def test_nearest_dynamic_class(report):
    nearest = report["traffic_context"]["nearest_dynamic_class"]
    assert nearest["class"] == "person"
    assert nearest["proximity"] == "near"


# 9. near/middle/far categorization
def test_region_labels_follow_median_depth():
    fr = fuse(SEG, DEPTH)
    low, high = fr.thresholds
    assert fr.per_class[11]["median_depth"] >= high  # person -> near
    assert fr.per_class[13]["median_depth"] < low  # car -> far


# 10. inverse-depth convention preserved in output
def test_depth_convention_is_explicit(report):
    assert report["scene"]["depth_convention"] == "inverse_relative_larger_closer"
    assert report["scene"]["depth_thresholds"]["high"] > report["scene"]["depth_thresholds"]["low"]


# 17. deterministic results
def test_results_are_deterministic():
    assert analyze_maps(SEG, DEPTH) == analyze_maps(SEG, DEPTH)


def test_interpretation_lines_present(report):
    assert isinstance(report["interpretation"], list)
    assert len(report["interpretation"]) > 0
    assert "relative-depth region" in report["interpretation"][0]


# 18. same-image shape contract
def test_analyze_maps_shape_contract():
    bad = np.zeros((2, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="identical spatial dimensions"):
        analyze_maps(bad, DEPTH)


# Analyze a scene from stub predictors (same input to both models)
def test_analyze_scene_runs_through_predictors():
    sentinel = object()
    seg_p = _StubSegPredictor(SEG)
    depth_p = _StubDepthPredictor(DEPTH)
    result = analyze_scene(sentinel, seg_p, depth_p)
    assert seg_p.last_input is sentinel
    assert depth_p.last_input is sentinel
    assert result == analyze_maps(SEG, DEPTH)


def test_analyze_scene_accepts_torch_outputs():
    result = analyze_scene(object(), _StubSegPredictor(SEG), _StubDepthPredictor(DEPTH))
    assert result["scene"]["analyzed_pixels"] == 9


# config-driven roi classes
def test_roi_classes_from_config():
    cfg = {"fusion": {"roi_classes": [13]}}
    report = analyze_maps(SEG, DEPTH, cfg=cfg)
    assert list(report["traffic_context"]["vehicles"]) == ["car"]
    assert set(report["traffic_context"]["pedestrians"]) == {"person"}
    assert report["traffic_context"]["dynamic_object_count"] == 1


def test_roi_override_beats_config():
    cfg = {"fusion": {"roi_classes": [0]}}
    report = analyze_maps(SEG, DEPTH, cfg=cfg, roi_classes=[13])
    assert "car" in report["traffic_context"]["vehicles"]


# config quantiles change the categorization
def test_config_region_quantiles_change_labels():
    report = analyze_maps(SEG, DEPTH, cfg={"fusion": {"region_quantiles": [0.5, 0.75]}})
    assert report["traffic_context"]["road"]["road"]["proximity"] == "far"
    assert report["traffic_context"]["vehicles"]["car"]["proximity"] == "far"
    assert report["traffic_context"]["pedestrians"]["person"]["proximity"] == "near"


# missing roi classes -> no vehicle/pedestrian entries, no error
def test_no_dynamic_objects_present():
    seg = np.full((3, 3), 0, dtype=np.int64)  # road only
    depth = np.arange(9, dtype=np.float64).reshape(3, 3)
    report = analyze_maps(seg, depth)
    assert report["traffic_context"]["vehicles"] == {}
    assert report["traffic_context"]["pedestrians"] == {}
    assert report["traffic_context"]["nearest_dynamic_class"] is None
    assert report["traffic_context"]["dynamic_object_count"] == 0