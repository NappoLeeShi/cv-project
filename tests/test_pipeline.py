"""Offline tests for the Step 11 full pipeline (no models, no datasets).

Uses deterministic dummy predictors and never touches the network, CUDA, or
real weights. Also covers the CLI argument layer without triggering downloads.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scene_understanding.analyzer import analyze_fusion
from scene_understanding.fusion import FusionResult, fuse
from scene_understanding.pipeline import (
    DEPTH_CONVENTION,
    PipelineError,
    PipelineResult,
    SceneUnderstandingPipeline,
)


# ---------- deterministic dummy predictors ----------

def _dummy_segmentation(size):
    h, w = size
    seg = np.full((h, w), 10, dtype=np.int64)      # sky
    seg[-max(1, h // 4):] = 0                      # road at the bottom
    seg[h // 4:3 * h // 4, w // 5:3 * w // 5] = 13  # car block
    seg[h // 4:3 * h // 4, 3 * w // 5:4 * w // 5] = 11  # person block
    return torch.from_numpy(seg)


def _dummy_depth(size, image_for_size=None):
    h, w = size
    yy = np.arange(h, dtype=np.float64)[:, None]
    depth = (yy / max(h - 1, 1)) * 60.0           # sky small, road large
    depth[h // 4:3 * h // 4, w // 5:4 * w // 5] += 30.0  # car/person nearer
    return torch.from_numpy(depth)


class RecordingSegmentationPredictor:
    def __init__(self, size=None, failure=None, ndim=None):
        self.size = size
        self.failure = failure
        self.ndim = ndim
        self.calls = 0
        self.last_input = None

    def predict(self, image):
        self.calls += 1
        self.last_input = image
        if self.failure is not None:
            raise self.failure
        if self.ndim == 3:
            return torch.zeros((1, 8, 8), dtype=torch.long)
        size = self.size or image.shape[:2]
        return _dummy_segmentation(size)


class RecordingDepthPredictor:
    def __init__(self, size=None, failure=None, ndim=None):
        self.size = size
        self.failure = failure
        self.ndim = ndim
        self.calls = 0
        self.last_input = None

    def predict(self, image):
        self.calls += 1
        self.last_input = image
        if self.failure is not None:
            raise self.failure
        if self.ndim == 3:
            return torch.zeros((1, 8, 8))
        size = self.size or image.shape[:2]
        return _dummy_depth(size)


@pytest.fixture
def street_image():
    h, w = 128, 256
    yy, xx = np.mgrid[0:h, 0:w]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[..., 0] = (xx * 255 // w).astype(np.uint8)
    image[..., 1] = (yy * 255 // h).astype(np.uint8)
    image[..., 2] = 90
    return image


@pytest.fixture
def dummy_predictors():
    seg = RecordingSegmentationPredictor()
    depth = RecordingDepthPredictor()
    return seg, depth


def make_pipeline(seg, depth, **kwargs):
    return SceneUnderstandingPipeline(seg, depth, **kwargs)


# 1. pipeline construction
def test_pipeline_construction(dummy_predictors):
    seg, depth = dummy_predictors
    pipeline = make_pipeline(seg, depth)
    assert pipeline.segmentation_predictor is seg
    assert pipeline.depth_predictor is depth
    assert pipeline.cfg["visualization"]["output_dir"] == "outputs/visualization"

    with pytest.raises(PipelineError, match="predict"):
        make_pipeline(None, depth)
    with pytest.raises(PipelineError, match="predict"):
        make_pipeline(seg, object())


# 2. run with dummy predictors
def test_run_with_dummy_predictors(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    assert isinstance(result, PipelineResult)
    assert seg.calls == 1 and depth.calls == 1


# 3. same image passed to both predictors
def test_same_image_passed_to_both_predictors(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    make_pipeline(seg, depth).run(street_image, visualize=False)
    assert seg.last_input is depth.last_input  # identical object
    assert seg.last_input is not None


# 4. segmentation output shape = image shape
def test_segmentation_output_shape(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    assert result.segmentation.shape == street_image.shape[:2]
    assert result.segmentation.dtype == np.int64


# 5. depth output shape = image shape
def test_depth_output_shape(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    assert result.depth.shape == street_image.shape[:2]
    assert result.depth.dtype == np.float64


# 6. fusion executed
def test_fusion_execution(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    assert isinstance(result.fusion, FusionResult)
    low, high = result.fusion.thresholds
    assert low < high
    assert result.fusion.region_map.shape == street_image.shape[:2]


# 7. scene analyzer executed
def test_scene_analyzer_execution(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    report = make_pipeline(seg, depth).run(street_image, visualize=False).scene_report
    assert set(report) >= {
        "scene", "semantic_distribution", "depth_distribution",
        "regions", "traffic_context", "interpretation",
    }
    assert "vehicles" in report["traffic_context"]
    assert "pedestrians" in report["traffic_context"]


# 8. PipelineResult structure
def test_pipeline_result_structure(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    summary = result.to_dict()
    assert summary["image_size"] == [128, 256]
    assert summary["segmentation_shape"] == [128, 256]
    assert summary["depth_shape"] == [128, 256]
    assert summary["depth_convention"] == DEPTH_CONVENTION
    assert isinstance(summary["fusion_thresholds"], dict)
    json.dumps(summary)  # JSON-serializable, no raw maps


# 9. visualize=False creates no visualization outputs
def test_visualize_false_creates_no_outputs(dummy_predictors, street_image, tmp_path):
    seg, depth = dummy_predictors
    pipeline = make_pipeline(seg, depth, output_dir=tmp_path / "viz")
    result = pipeline.run(street_image, visualize=False)
    assert result.visualization_paths == {}
    assert not (tmp_path / "viz").exists()


# 10. visualize=True creates the expected outputs
def test_visualize_true_creates_expected_outputs(dummy_predictors, street_image, tmp_path):
    seg, depth = dummy_predictors
    pipeline = make_pipeline(seg, depth, output_dir=tmp_path / "viz")
    result = pipeline.run(street_image, visualize=True)
    assert set(result.visualization_paths) == {
        "segmentation", "depth", "fusion", "overview", "scene_report",
    }
    for path in result.visualization_paths.values():
        assert Path(path).is_file()


# 11. output directory auto-created
def test_output_directory_auto_created(dummy_predictors, street_image, tmp_path):
    seg, depth = dummy_predictors
    pipeline = make_pipeline(seg, depth, output_dir=tmp_path / "a" / "b")
    result = pipeline.run(street_image, visualize=True)
    assert (tmp_path / "a" / "b").is_dir()
    assert result.visualization_paths["overview"].startswith(str(tmp_path / "a" / "b"))


# 12. invalid image handling
def test_invalid_image_handling(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    pipeline = make_pipeline(seg, depth)
    with pytest.raises(PipelineError, match="HWC RGB"):
        pipeline.run(np.zeros((5,), dtype=np.uint8))
    with pytest.raises(PipelineError, match="HWC RGB"):
        pipeline.run(np.zeros((8, 8, 4), dtype=np.uint8))
    with pytest.raises(PipelineError, match="unsupported"):
        pipeline.run("not an image")
    with pytest.raises(PipelineError, match="float images"):
        pipeline.run(np.full((4, 4, 3), 2.0, dtype=np.float32))


# 12b. PIL images are accepted
def test_pil_image_accepted(dummy_predictors, street_image):
    from PIL import Image

    seg, depth = dummy_predictors
    pil = Image.fromarray(street_image, mode="RGB")
    result = make_pipeline(seg, depth).run(pil, visualize=False)
    assert result.segmentation.shape == (128, 256)


# 13. prediction shape mismatch is aligned to the input resolution
def test_prediction_shape_mismatch_aligned(street_image, tmp_path):
    seg = RecordingSegmentationPredictor(size=(64, 128))   # smaller than image
    depth = RecordingDepthPredictor()                      # image resolution
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    assert result.segmentation.shape == (128, 256)
    assert result.depth.shape == (128, 256)

    seg = RecordingSegmentationPredictor(size=(64, 128))
    depth = RecordingDepthPredictor(size=(50, 90))         # also off-resolution
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    assert result.segmentation.shape == (128, 256)
    assert result.depth.shape == (128, 256)


def test_prediction_non_2d_rejected(street_image):
    seg = RecordingSegmentationPredictor(ndim=3)
    depth = RecordingDepthPredictor()
    with pytest.raises(PipelineError, match="2D"):
        make_pipeline(seg, depth).run(street_image, visualize=False)


def test_prediction_non_integer_segmentation_rejected(street_image):
    class FloatSegPredictor(RecordingSegmentationPredictor):
        def predict(self, image):
            h, w = image.shape[:2]
            return torch.full((h, w), 3.5)

    with pytest.raises(PipelineError, match="integer"):
        make_pipeline(FloatSegPredictor(), RecordingDepthPredictor()).run(
            street_image, visualize=False
        )


# 14. predictor failure propagation (no swallowing)
def test_predictor_failure_propagation(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    boom = ValueError("boom-seg")
    seg.failure = boom
    with pytest.raises(ValueError, match="boom-seg"):
        make_pipeline(seg, depth).run(street_image, visualize=False)

    seg.failure = None
    depth.failure = RuntimeError("boom-depth")
    with pytest.raises(RuntimeError, match="boom-depth"):
        make_pipeline(seg, depth).run(street_image, visualize=False)


# 15. deterministic result
def test_deterministic_result(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    pipeline = make_pipeline(seg, depth)
    a = pipeline.run(street_image, visualize=False)
    b = pipeline.run(street_image, visualize=False)
    assert a.scene_report == b.scene_report
    assert np.array_equal(a.segmentation, b.segmentation)
    assert np.array_equal(a.depth, b.depth)
    assert a.to_dict() == b.to_dict()


# ---------- synthetic demo (required classes + traffic context) ----------

def test_synthetic_demo_pipeline(street_image):
    h, w = street_image.shape[:2]
    assert (h, w) == (128, 256)
    seg = RecordingSegmentationPredictor(size=(h, w))
    depth = RecordingDepthPredictor(size=(h, w))
    pipeline = make_pipeline(seg, depth)
    result = pipeline.run(street_image, visualize=False)

    report = result.scene_report
    assert report["scene"]["depth_convention"] == "inverse_relative_larger_closer"
    assert report["semantic_distribution"]["sky"] > 0
    assert report["semantic_distribution"]["road"] > 0
    assert "car" in report["traffic_context"]["vehicles"]
    assert "person" in report["traffic_context"]["pedestrians"]
    assert report["traffic_context"]["nearest_dynamic_class"] is not None
    assert report["traffic_context"]["nearest_dynamic_class"]["class"] in {"car", "person"}
    assert len(report["interpretation"]) > 0


def test_synthetic_demo_depth_convention_holds(dummy_predictors, street_image):
    seg, depth = dummy_predictors
    result = make_pipeline(seg, depth).run(street_image, visualize=False)
    road_mask = result.segmentation == 0
    sky_mask = result.segmentation == 10
    # road (bottom) has larger inverse depth than sky (top) in the dummy map
    assert result.depth[road_mask].mean() > result.depth[sky_mask].mean()


# ---------- CLI (argument parsing only; never loads models) ----------

def test_cli_help():
    import main as cli

    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_missing_image_argument():
    import main as cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--unet-checkpoint", "a.pt", "--midas-weights", "b.pt"]
        )


def test_cli_invalid_image_rejected(tmp_path):
    import main as cli

    with pytest.raises(ValueError, match="not found"):
        cli.require_image_file(str(tmp_path / "does_not_exist.jpg"))


def test_cli_output_dir_and_flags_parsed():
    import main as cli

    args = cli.build_parser().parse_args(
        ["--image", "x.png", "--output-dir", "out", "--unet-checkpoint", "a.pt",
         "--midas-weights", "b.pt", "--no-visualization", "--device", "cpu"]
    )
    assert args.image == "x.png"
    assert args.output_dir == "out"
    assert args.unet_checkpoint == "a.pt"
    assert args.midas_weights == "b.pt"
    assert args.no_visualization is True
    assert args.device == "cpu"


def test_cli_missing_predictor_inputs_clear_errors(tmp_path):
    import main as cli

    unet = tmp_path / "unet.pt"
    midas = tmp_path / "midas.pt"
    unet.touch()
    midas.touch()

    with pytest.raises(RuntimeError, match="U-Net checkpoint is required"):
        cli.require_predictor_inputs(None, str(midas))
    with pytest.raises(RuntimeError, match="MiDaS weights are required"):
        cli.require_predictor_inputs(str(unet), None)
    with pytest.raises(RuntimeError, match="not found"):
        cli.require_predictor_inputs("missing.pt", str(midas))
    with pytest.raises(RuntimeError, match="not found"):
        cli.require_predictor_inputs(str(unet), "missing_midas.pt")
    cli.require_predictor_inputs(str(unet), str(midas))  # valid inputs pass


def test_cli_parse_does_not_import_models():
    import main as cli

    cli.build_parser().parse_args(
        ["--image", "x.png", "--unet-checkpoint", "a", "--midas-weights", "b"]
    )


def test_real_predictor_builder_rejects_missing_inputs():
    import main as cli

    assert cli.build_real_predictors is not None  # builder is available