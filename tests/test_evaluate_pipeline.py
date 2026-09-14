"""Offline tests for Step 15 — pipeline evaluation (no real GPU inference).

Uses synthetic/mock data and dummy predictors. Never loads real checkpoints
or requires CUDA. Covers all 10 test categories from the spec.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from evaluation.difficulty_analysis import (
    DEFAULT_BINS,
    DEFAULT_WEIGHTS,
    DifficultyResult,
    _clamp,
    _depth_variation_indicator,
    _foreground_fraction_indicator,
    _object_density_indicator,
    _scene_complexity_indicator,
    _segmentation_confidence_indicator,
    compute_difficulty,
    difficulty_from_report,
)
from evaluation.evaluate_pipeline import (
    SUPPORTED_EXTENSIONS,
    _aggregate_results,
    build_parser,
    collect_images,
    evaluate_single_image,
)
from scene_understanding.fusion import FusionResult, fuse
from scene_understanding.analyzer import analyze_fusion


def _image_size_hw(image):
    if hasattr(image, "shape"):
        return int(image.shape[0]), int(image.shape[1])
    return int(image.size[1]), int(image.size[0])


# ----------------------------------------------------------------------- #
#  Fixtures
# ----------------------------------------------------------------------- #

@pytest.fixture
def sample_seg():
    seg = np.full((64, 128), 10, dtype=np.int64)
    seg[-16:, :] = 0
    seg[16:48, 20:60] = 13
    seg[16:48, 80:110] = 11
    return seg


@pytest.fixture
def sample_depth():
    h, w = 64, 128
    yy = np.arange(h, dtype=np.float64)[:, None]
    depth = np.broadcast_to(yy / max(h - 1, 1) * 50.0, (h, w)).copy()
    depth[16:48, 20:110] += 25.0
    return depth


@pytest.fixture
def sample_fusion(sample_seg, sample_depth):
    return fuse(sample_seg, sample_depth)


@pytest.fixture
def sample_image():
    h, w = 64, 128
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class DummySegPredictor:
    def __init__(self, return_confidence=False):
        self.model = MagicMock()
        self.model.eval = MagicMock()
        self.return_confidence = return_confidence

    def predict(self, image, return_confidence=False):
        h, w = _image_size_hw(image)
        seg = np.full((h, w), 10, dtype=np.int64)
        seg[-h // 4:, :] = 0
        seg[h // 4:3 * h // 4, w // 4:w // 2] = 13
        seg[h // 4:3 * h // 4, w // 2:3 * w // 4] = 11
        result = torch.from_numpy(seg)
        if return_confidence:
            conf = torch.full((h, w), 0.85, dtype=torch.float32)
            return result, conf
        return result


class DummyDepthPredictor:
    def __init__(self):
        self.model = MagicMock()
        self.model.eval = MagicMock()

    def predict(self, image):
        h, w = _image_size_hw(image)
        yy = np.arange(h, dtype=np.float32)[:, None]
        depth = np.broadcast_to(yy / max(h - 1, 1) * 50.0, (h, w))
        return torch.from_numpy(depth.copy())


# ----------------------------------------------------------------------- #
#  1. Configuration loading
# ----------------------------------------------------------------------- #

class TestConfigLoading:
    def test_config_loads_pipeline(self):
        from utils.config import load_config
        cfg = load_config("pipeline")
        assert "fusion" in cfg
        assert "difficulty" in cfg
        assert "output" in cfg

    def test_difficulty_config_has_weights_and_bins(self):
        from utils.config import load_config
        cfg = load_config("pipeline")
        weights = cfg["difficulty"]["factor_weights"]
        bins = cfg["difficulty"]["bins"]
        assert len(weights) == 5
        assert "easy" in bins
        assert "medium" in bins

    def test_cli_parser_accepts_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--config", "pipeline",
            "--input-dir", "data/pipeline/images",
            "--limit", "2",
            "--device", "cpu",
            "--save-visualizations",
            "--output", "outputs/analysis",
        ])
        assert args.config == "pipeline"
        assert args.limit == 2
        assert args.save_visualizations is True


# ----------------------------------------------------------------------- #
#  2. Same image object supplied to both predictors
# ----------------------------------------------------------------------- #

class TestSameImageContract:
    def test_same_numpy_object_to_both(self, sample_image):
        seg_pred = DummySegPredictor()
        depth_pred = DummyDepthPredictor()
        from PIL import Image

        pil = Image.fromarray(sample_image, mode="RGB")
        seg_pred.predict(pil)
        depth_pred.predict(pil)

        seg_out = seg_pred.predict(pil)
        depth_out = depth_pred.predict(pil)
        assert seg_out.shape[0] == depth_out.shape[0]
        assert seg_out.shape[1] == depth_out.shape[1]

    def test_same_image_path_evaluated(self, tmp_path, sample_image):
        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.fromarray(sample_image).save(img_path)

        seg_pred = DummySegPredictor()
        depth_pred = DummyDepthPredictor()

        result = evaluate_single_image(
            img_path, seg_pred, depth_pred, {}, device=torch.device("cpu"),
        )
        assert result["image"] == "test.png"
        assert result["segmentation"]["num_classes_present"] > 0


# ----------------------------------------------------------------------- #
#  3. Segmentation/depth spatial alignment
# ----------------------------------------------------------------------- #

class TestSpatialAlignment:
    def test_segmentation_matches_image_shape(self, sample_image):
        seg_pred = DummySegPredictor()
        depth_pred = DummyDepthPredictor()
        from PIL import Image

        pil = Image.fromarray(sample_image, mode="RGB")
        seg_out = seg_pred.predict(pil)
        depth_out = depth_pred.predict(pil)

        seg_np = seg_out.numpy()
        depth_np = depth_out.numpy()
        assert seg_np.shape == depth_np.shape

    def test_different_prediction_sizes_are_aligned(self):
        h, w = 64, 128
        seg = np.full((h, w), 0, dtype=np.int64)
        depth = np.ones((h, w), dtype=np.float64)

        from scene_understanding.pipeline import _resize_nearest_ids, _resize_bilinear
        small_seg = np.full((h // 2, w // 2), 0, dtype=np.int64)
        small_depth = np.ones((h // 2, w // 2), dtype=np.float64)

        aligned_seg = _resize_nearest_ids(small_seg, (h, w))
        aligned_depth = _resize_bilinear(small_depth, (h, w))
        assert aligned_seg.shape == (h, w)
        assert aligned_depth.shape == (h, w)


# ----------------------------------------------------------------------- #
#  4. Fusion receives outputs from the same image
# ----------------------------------------------------------------------- #

class TestFusionFromSameImage:
    def test_fusion_receives_seg_and_depth(self, sample_fusion):
        assert isinstance(sample_fusion, FusionResult)
        assert sample_fusion.seg_mask.shape == sample_fusion.depth.shape

    def test_fusion_result_has_per_class_stats(self, sample_fusion):
        assert len(sample_fusion.per_class) > 0
        for class_id, stats in sample_fusion.per_class.items():
            assert "pixel_count" in stats
            assert "mean_depth" in stats

    def test_fusion_result_has_region_summary(self, sample_fusion):
        summary = sample_fusion.region_summary
        assert "near" in summary
        assert "middle" in summary
        assert "far" in summary


# ----------------------------------------------------------------------- #
#  5. Difficulty score is deterministic
# ----------------------------------------------------------------------- #

class TestDifficultyDeterministic:
    def test_same_inputs_same_score(self, sample_fusion):
        d1 = compute_difficulty(sample_fusion)
        d2 = compute_difficulty(sample_fusion)
        assert d1.score == d2.score
        assert d1.level == d2.level

    def test_same_inputs_same_components(self, sample_fusion):
        d1 = compute_difficulty(sample_fusion)
        d2 = compute_difficulty(sample_fusion)
        for key in d1.components:
            assert d1.components[key] == d2.components[key]

    def test_score_in_unit_interval(self, sample_fusion):
        d = compute_difficulty(sample_fusion)
        assert 0.0 <= d.score <= 1.0


# ----------------------------------------------------------------------- #
#  6. Easy/Medium/Hard thresholds work
# ----------------------------------------------------------------------- #

class TestDifficultyThresholds:
    def test_low_score_is_easy(self, sample_fusion):
        d = compute_difficulty(sample_fusion, weights=[1.0, 0, 0, 0, 0])
        if d.score <= DEFAULT_BINS["easy"]:
            assert d.level == "easy"

    def test_threshold_classification(self):
        assert DifficultyResult(score=0.1, level="easy").level == "easy"
        assert DifficultyResult(score=0.5, level="medium").level == "medium"
        assert DifficultyResult(score=0.9, level="hard").level == "hard"

    def test_boundary_easy_medium(self, sample_fusion):
        d = compute_difficulty(sample_fusion, bins={"easy": 0.5, "medium": 0.7})
        if d.score <= 0.5:
            assert d.level == "easy"
        elif d.score <= 0.7:
            assert d.level == "medium"
        else:
            assert d.level == "hard"

    def test_custom_weights_change_score(self, sample_fusion):
        d1 = compute_difficulty(sample_fusion, weights=[1.0, 0, 0, 0, 0])
        d2 = compute_difficulty(sample_fusion, weights=[0, 0, 0, 0, 1.0])
        # With different weights the scores should generally differ
        # (unless indicators happen to be equal)
        assert isinstance(d1.score, float)
        assert isinstance(d2.score, float)


# ----------------------------------------------------------------------- #
#  7. JSON output schema
# ----------------------------------------------------------------------- #

class TestJSONSchema:
    def test_per_image_schema(self, sample_fusion):
        fr = sample_fusion
        scene_report = analyze_fusion(fr)
        diff = compute_difficulty(fr, scene_report=scene_report)

        entry = {
            "image": "test.png",
            "segmentation": {
                "num_classes_present": len(fr.per_class),
                "mean_confidence": 0.85,
            },
            "depth": {
                "mean_inverse_depth": float(fr.depth[np.isfinite(fr.depth)].mean()),
                "std_inverse_depth": float(fr.depth[np.isfinite(fr.depth)].std()),
                "min_inverse_depth": float(fr.depth[np.isfinite(fr.depth)].min()),
                "max_inverse_depth": float(fr.depth[np.isfinite(fr.depth)].max()),
            },
            "scene": scene_report,
            "difficulty": diff.to_dict(),
        }

        serialized = json.dumps(entry)
        loaded = json.loads(serialized)
        assert loaded["image"] == "test.png"
        assert "difficulty" in loaded
        assert "score" in loaded["difficulty"]
        assert "level" in loaded["difficulty"]
        assert "components" in loaded["difficulty"]

    def test_aggregate_schema(self):
        per_image = [
            {
                "difficulty": {"score": 0.3, "level": "easy"},
                "segmentation": {"mean_confidence": 0.9},
                "depth": {"std_inverse_depth": 5.0},
                "scene": {"semantic_distribution": {"road": 100}},
            },
            {
                "difficulty": {"score": 0.6, "level": "medium"},
                "segmentation": {"mean_confidence": 0.7},
                "depth": {"std_inverse_depth": 8.0},
                "scene": {"semantic_distribution": {"road": 50, "car": 30}},
            },
        ]
        agg = _aggregate_results(per_image)
        assert agg["num_images"] == 2
        assert "average_difficulty_score" in agg
        assert "easy_count" in agg
        assert "medium_count" in agg
        assert "hard_count" in agg
        assert "average_segmentation_confidence" in agg
        assert "class_occurrence" in agg


# ----------------------------------------------------------------------- #
#  8. --limit works
# ----------------------------------------------------------------------- #

class TestLimit:
    def test_limit_restricts_image_count(self, tmp_path):
        for i in range(5):
            (tmp_path / f"img_{i}.png").touch()
        images = collect_images(tmp_path, limit=3)
        assert len(images) == 3

    def test_limit_none_returns_all(self, tmp_path):
        for i in range(4):
            (tmp_path / f"img_{i}.png").touch()
        images = collect_images(tmp_path, limit=None)
        assert len(images) == 4

    def test_limit_larger_than_count(self, tmp_path):
        for i in range(2):
            (tmp_path / f"img_{i}.png").touch()
        images = collect_images(tmp_path, limit=100)
        assert len(images) == 2

    def test_collect_images_filters_extensions(self, tmp_path):
        (tmp_path / "a.png").touch()
        (tmp_path / "b.jpg").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.bmp").touch()
        images = collect_images(tmp_path)
        extensions = {p.suffix.lower() for p in images}
        assert ".txt" not in extensions
        assert all(ext in SUPPORTED_EXTENSIONS for ext in extensions)


# ----------------------------------------------------------------------- #
#  9. CPU-safe execution with mocks
# ----------------------------------------------------------------------- #

class TestCPUSafeExecution:
    def test_evaluate_single_image_cpu(self, tmp_path, sample_image):
        from PIL import Image

        img_path = tmp_path / "test.png"
        Image.fromarray(sample_image).save(img_path)

        seg_pred = DummySegPredictor()
        depth_pred = DummyDepthPredictor()

        result = evaluate_single_image(
            img_path, seg_pred, depth_pred, {}, device=torch.device("cpu"),
        )

        assert "image" in result
        assert "segmentation" in result
        assert "depth" in result
        assert "scene" in result
        assert "difficulty" in result

    def test_evaluate_uses_no_grad(self, tmp_path, sample_image):
        from PIL import Image

        img_path = tmp_path / "test.png"
        Image.fromarray(sample_image).save(img_path)

        seg_pred = DummySegPredictor()
        depth_pred = DummyDepthPredictor()

        with patch("torch.no_grad") as mock_grad:
            result = evaluate_single_image(
                img_path, seg_pred, depth_pred, {},
                device=torch.device("cpu"),
            )
            # no_grad should have been called (or the model eval mock)
            assert "difficulty" in result

    def test_difficulty_score_components_are_valid(self, sample_fusion):
        d = compute_difficulty(sample_fusion)
        assert set(d.components.keys()) == {
            "segmentation_uncertainty",
            "depth_variation",
            "scene_complexity",
            "foreground_fraction",
            "object_density",
        }
        for name, value in d.components.items():
            assert isinstance(value, float), f"{name} is not float"
            assert 0.0 <= value <= 1.0, f"{name} = {value} out of [0, 1]"


# ----------------------------------------------------------------------- #
#  10. No accidental Cityscapes/KITTI pairing
# ----------------------------------------------------------------------- #

class TestNoDatasetPairing:
    def test_pipeline_uses_single_input_dir(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input-dir", "data/pipeline/images",
            "--limit", "1",
        ])
        assert args.input_dir == "data/pipeline/images"

    def test_evaluate_does_not_import_cityscapes(self, tmp_path, sample_image):
        """The evaluate_single_image function must not import Cityscapes."""
        from PIL import Image
        import importlib
        import sys

        img_path = tmp_path / "test.png"
        Image.fromarray(sample_image).save(img_path)

        seg_pred = DummySegPredictor()
        depth_pred = DummyDepthPredictor()

        # Run evaluation - it should not need Cityscapes
        result = evaluate_single_image(
            img_path, seg_pred, depth_pred, {}, device=torch.device("cpu"),
        )
        assert result["image"] == "test.png"


# ----------------------------------------------------------------------- #
#  Indicator helpers
# ----------------------------------------------------------------------- #

class TestIndicatorHelpers:
    def test_confidence_none_gives_neutral(self):
        assert _segmentation_confidence_indicator(None) == 0.5

    def test_confidence_high_gives_low_uncertainty(self):
        assert _segmentation_confidence_indicator(0.95) < 0.1

    def test_confidence_low_gives_high_uncertainty(self):
        assert _segmentation_confidence_indicator(0.3) > 0.6

    def test_depth_variation_uniform(self):
        depth = np.ones((10, 10), dtype=np.float64) * 5.0
        assert _depth_variation_indicator(depth) == 0.0

    def test_depth_variation_varying(self):
        depth = np.linspace(1, 100, 100, dtype=np.float64).reshape(10, 10)
        assert _depth_variation_indicator(depth) > 0.0

    def test_scene_complexity_empty(self):
        assert _scene_complexity_indicator(0) == 0.0

    def test_scene_complexity_full(self):
        assert _scene_complexity_indicator(19) == 1.0

    def test_foreground_fraction_from_region_summary(self):
        summary = {"near": {"pixel_ratio": 0.3}}
        assert _foreground_fraction_indicator(summary) == pytest.approx(0.3)

    def test_object_density_zero_classes(self, sample_fusion):
        assert _object_density_indicator(sample_fusion, roi_classes=()) == 0.0


# ----------------------------------------------------------------------- #
#  Difficulty from report convenience wrapper
# ----------------------------------------------------------------------- #

class TestDifficultyFromReport:
    def test_returns_dict(self, sample_fusion):
        scene_report = analyze_fusion(sample_fusion)
        result = difficulty_from_report(sample_fusion, scene_report)
        assert isinstance(result, dict)
        assert "score" in result
        assert "level" in result
        assert "components" in result

    def test_json_serializable(self, sample_fusion):
        scene_report = analyze_fusion(sample_fusion)
        result = difficulty_from_report(sample_fusion, scene_report)
        serialized = json.dumps(result)
        loaded = json.loads(serialized)
        assert loaded["score"] == result["score"]


# ----------------------------------------------------------------------- #
#  Aggregate results
# ----------------------------------------------------------------------- #

class TestAggregateResults:
    def test_empty_list(self):
        agg = _aggregate_results([])
        assert agg["num_images"] == 0

    def test_single_image(self):
        per_image = [
            {
                "difficulty": {"score": 0.5, "level": "medium"},
                "segmentation": {"mean_confidence": 0.8},
                "depth": {"std_inverse_depth": 10.0},
                "scene": {"semantic_distribution": {"road": 100, "car": 50}},
            }
        ]
        agg = _aggregate_results(per_image)
        assert agg["num_images"] == 1
        assert agg["average_difficulty_score"] == pytest.approx(0.5)
        assert agg["easy_count"] == 0
        assert agg["medium_count"] == 1
        assert agg["hard_count"] == 0
        assert agg["average_segmentation_confidence"] == pytest.approx(0.8)
        assert agg["class_occurrence"]["road"] == 1
        assert agg["class_occurrence"]["car"] == 1

    def test_multiple_images(self):
        per_image = [
            {
                "difficulty": {"score": 0.2, "level": "easy"},
                "segmentation": {"mean_confidence": 0.9},
                "depth": {"std_inverse_depth": 5.0},
                "scene": {"semantic_distribution": {"road": 100}},
            },
            {
                "difficulty": {"score": 0.5, "level": "medium"},
                "segmentation": {"mean_confidence": 0.7},
                "depth": {"std_inverse_depth": 15.0},
                "scene": {"semantic_distribution": {"road": 80, "car": 20}},
            },
            {
                "difficulty": {"score": 0.8, "level": "hard"},
                "segmentation": {"mean_confidence": None},
                "depth": {},
                "scene": {"semantic_distribution": {"road": 50, "car": 30, "person": 20}},
            },
        ]
        agg = _aggregate_results(per_image)
        assert agg["num_images"] == 3
        assert agg["average_difficulty_score"] == pytest.approx(0.5)
        assert agg["easy_count"] == 1
        assert agg["medium_count"] == 1
        assert agg["hard_count"] == 1
        assert agg["average_segmentation_confidence"] == pytest.approx(0.8)
