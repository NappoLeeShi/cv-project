"""Offline tests for Step 17 — Custom Image Fusion Demo.

Uses synthetic images and mocked predictors; never loads the real MiDaS
checkpoint (1.4 GB) or the U-Net checkpoint. Also keeps a regression guard on
the Step 11 single-image CLI and Step 15 pipeline helpers.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from evaluation.custom_demo import (
    SUPPORTED_EXTENSIONS,
    CustomDemoError,
    build_scene_report,
    discover_images,
    load_rgb_image,
    run_custom_demo,
    validate_checkpoints,
)
from utils.config import load_config


def _image_size_hw(image):
    if hasattr(image, "shape"):
        return int(image.shape[0]), int(image.shape[1])
    return int(image.size[1]), int(image.size[0])


# ----------------------------------------------------------------------- #
#  Deterministic dummy predictors (record the exact input object)
# ----------------------------------------------------------------------- #

class DummySegPredictor:
    def __init__(self):
        self.model = MagicMock()
        self.model.eval = MagicMock()
        self.last_input = None
        self.inputs = []

    def predict(self, image, return_confidence=False):
        self.last_input = image
        self.inputs.append(image)
        h, w = _image_size_hw(image)
        seg = np.full((h, w), 10, dtype=np.int64)
        seg[-h // 4:, :] = 0
        seg[h // 4:3 * h // 4, w // 4:w // 2] = 13
        seg[h // 4:3 * h // 4, w // 2:3 * w // 4] = 11
        result = torch.from_numpy(seg)
        if return_confidence:
            confidence = torch.full((h, w), 0.85, dtype=torch.float32)
            return result, confidence
        return result


class DummyDepthPredictor:
    def __init__(self):
        self.model = MagicMock()
        self.model.eval = MagicMock()
        self.last_input = None
        self.inputs = []

    def predict(self, image):
        self.last_input = image
        self.inputs.append(image)
        h, w = _image_size_hw(image)
        yy = np.arange(h, dtype=np.float32)[:, None]
        depth = np.broadcast_to(yy / max(h - 1, 1) * 50.0, (h, w))
        return torch.from_numpy(depth.copy())


def _make_image(path: Path, grayscale: bool = False, size=(48, 96), corrupt=False):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        path.write_bytes(b"this is not a valid image")
        return
    h, w = size
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    if grayscale:
        arr = arr.mean(axis=2).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


# ----------------------------------------------------------------------- #
#  CLI integration (main.py)
# ----------------------------------------------------------------------- #

class TestMainCLI:
    def test_input_dir_flag_parses(self):
        import main as cli

        args = cli.build_parser().parse_args(
            ["--input-dir", "data/pipeline/images", "--limit", "1", "--device", "cpu"]
        )
        assert args.input_dir == "data/pipeline/images"
        assert args.limit == 1
        assert args.device == "cpu"

    def test_image_mode_still_parses(self):
        import main as cli

        args = cli.build_parser().parse_args(
            ["--image", "x.png", "--unet-checkpoint", "a.pt",
             "--midas-weights", "b.pt"]
        )
        assert args.image == "x.png"
        assert args.unet_checkpoint == "a.pt"

    def test_both_modes_rejected(self):
        import main as cli

        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["--image", "x.png", "--input-dir", "images"]
            )

    def test_neither_mode_rejected(self):
        import main as cli

        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--limit", "1"])


# ----------------------------------------------------------------------- #
#  1. Input directory exists
# ----------------------------------------------------------------------- #

class TestInputDirectory:
    def test_missing_input_dir_raises_clear_error(self, tmp_path):
        seg, depth = DummySegPredictor(), DummyDepthPredictor()
        with pytest.raises(CustomDemoError, match="input directory not found"):
            run_custom_demo(
                str(tmp_path / "missing"), seg_predictor=seg, depth_predictor=depth,
            )


# ----------------------------------------------------------------------- #
#  2. Empty input directory handled
# ----------------------------------------------------------------------- #

class TestEmptyDirectory:
    def test_empty_input_dir_raises_clear_error(self, tmp_path):
        seg, depth = DummySegPredictor(), DummyDepthPredictor()
        with pytest.raises(CustomDemoError, match="no supported images"):
            run_custom_demo(
                str(tmp_path), seg_predictor=seg, depth_predictor=depth,
            )

    def test_unsupported_only_dir_raises_clear_error(self, tmp_path):
        (tmp_path / "notes.txt").write_text("nothing")
        with pytest.raises(CustomDemoError, match="no supported images"):
            run_custom_demo(
                str(tmp_path),
                seg_predictor=DummySegPredictor(),
                depth_predictor=DummyDepthPredictor(),
            )


# ----------------------------------------------------------------------- #
#  3 + 4. Supported extensions discovered / unsupported ignored
# ----------------------------------------------------------------------- #

class TestDiscovery:
    def test_supported_extensions_are_discovered(self, tmp_path):
        names = ["a.jpg", "b.jpeg", "c.png", "d.bmp", "e.webp", "f.tiff"]
        for name in names:
            _make_image(tmp_path / name)
        (tmp_path / "note.txt").write_text("skip me")
        (tmp_path / "junk").mkdir()

        images = discover_images(tmp_path)
        assert len(images) == len(names)
        extensions = {p.suffix.lower() for p in images}
        assert extensions == set(SUPPORTED_EXTENSIONS)

    def test_unsupported_files_ignored_in_run(self, tmp_path):
        _make_image(tmp_path / "street1.png")
        (tmp_path / "notes.txt").write_text("skip")
        seg, depth = DummySegPredictor(), DummyDepthPredictor()

        summary = run_custom_demo(
            str(tmp_path), seg_predictor=seg, depth_predictor=depth,
            output_dir=str(tmp_path / "out"),
        )
        assert summary["num_images"] == 1
        assert summary["images"][0]["filename"] == "street1.png"


# ----------------------------------------------------------------------- #
#  5 + 6. --limit and multiple images
# ----------------------------------------------------------------------- #

class TestLimit:
    @pytest.fixture
    def three_images(self, tmp_path):
        for i in range(3):
            _make_image(tmp_path / f"street{i}.png")
        return tmp_path

    def test_limit_one_processes_single_image(self, three_images, tmp_path):
        out = tmp_path / "out"
        run_custom_demo(
            str(three_images), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(), output_dir=str(out), limit=1,
        )
        reports = sorted(out.glob("*_scene_report.json"))
        assert len(reports) == 1
        assert reports[0].name == "street0_scene_report.json"

    def test_multiple_images_processed(self, three_images, tmp_path):
        out = tmp_path / "out"
        summary = run_custom_demo(
            str(three_images), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(), output_dir=str(out),
        )
        assert summary["num_images"] == 3
        assert len(list(out.glob("*_scene_report.json"))) == 3


# ----------------------------------------------------------------------- #
#  7 + 8. Output directory created + expected filenames
# ----------------------------------------------------------------------- #

EXPECTED_TAIL = ("_original.png", "_segmentation.png", "_depth.png",
                 "_fusion.png", "_overview.png", "_scene_report.json")


class TestOutputs:
    def test_output_directory_auto_created(self, tmp_path):
        _make_image(tmp_path / "street1.png")
        out = tmp_path / "a" / "b" / "c"
        assert not out.exists()
        run_custom_demo(
            str(tmp_path), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(), output_dir=str(out),
        )
        assert out.is_dir()

    def test_expected_output_filenames(self, tmp_path):
        _make_image(tmp_path / "street1.png")
        out = tmp_path / "out"
        run_custom_demo(
            str(tmp_path), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(), output_dir=str(out),
        )
        for suffix in EXPECTED_TAIL:
            assert (out / f"street1{suffix}").is_file(), f"missing street1{suffix}"

    def test_default_output_dir_is_outputs_custom(self, tmp_path, monkeypatch):
        _make_image(tmp_path / "street1.png")

        def fake_resolve(path):
            p = Path(path)
            return p if p.is_absolute() else tmp_path / "default_out"

        monkeypatch.setattr("evaluation.custom_demo.resolve_path", fake_resolve)
        summary = run_custom_demo(
            str(tmp_path), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(),
        )
        assert Path(summary["output_dir"]) == tmp_path / "default_out"
        assert (tmp_path / "default_out" / "street1_scene_report.json").is_file()


# ----------------------------------------------------------------------- #
#  9. Same image passed to both model predictors
# ----------------------------------------------------------------------- #

class TestSameImage:
    def test_same_image_object_to_both_predictors(self, tmp_path):
        _make_image(tmp_path / "street1.png")
        seg, depth = DummySegPredictor(), DummyDepthPredictor()
        run_custom_demo(
            str(tmp_path), seg_predictor=seg, depth_predictor=depth,
            output_dir=str(tmp_path / "out"),
        )
        assert seg.last_input is not None
        assert seg.last_input is depth.last_input  # identical ndarray object
        assert seg.inputs[0] is depth.inputs[0]


# ----------------------------------------------------------------------- #
#  10. Scene report JSON generated (rich content)
# ----------------------------------------------------------------------- #

class TestSceneReport:
    def test_scene_report_json_generated(self, tmp_path):
        _make_image(tmp_path / "street1.png")
        out = tmp_path / "out"
        run_custom_demo(
            str(tmp_path), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(), output_dir=str(out),
        )
        report_path = out / "street1_scene_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert set(report) >= {
            "image", "segmentation", "depth", "fusion", "scene", "difficulty",
        }
        assert report["image"]["filename"] == "street1.png"
        assert report["image"]["mode"] == "RGB"
        assert report["depth"]["convention"] == "inverse_relative_larger_closer"
        assert report["fusion"]["depth_convention"] == "inverse_relative_larger_closer"
        assert {"far", "middle", "near"} <= set(report["fusion"]["depth_regions"])
        assert "low" in report["fusion"]["thresholds"]
        assert "high" in report["fusion"]["thresholds"]
        assert report["scene"]["scene"]["depth_convention"] == "inverse_relative_larger_closer"
        assert "traffic_context" in report["scene"]
        assert report["difficulty"]["score"] >= 0.0
        assert report["difficulty"]["level"] in {"easy", "medium", "hard"}
        assert set(report["difficulty"]["components"]) == {
            "segmentation_uncertainty",
            "depth_variation",
            "scene_complexity",
            "foreground_fraction",
            "object_density",
        }

    def test_no_metrics_against_ground_truth(self, tmp_path):
        _make_image(tmp_path / "street1.png")
        out = tmp_path / "out"
        run_custom_demo(
            str(tmp_path), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(), output_dir=str(out),
        )
        report = json.loads((out / "street1_scene_report.json").read_text("utf-8"))
        for forbidden in ("absrel", "rmse", "mae", "iou", "pixel_accuracy"):
            assert forbidden not in report["difficulty"], forbidden
            assert forbidden not in report["depth"], forbidden


# ----------------------------------------------------------------------- #
#  Validation: grayscale / corrupt / checkpoints
# ----------------------------------------------------------------------- #

class TestValidation:
    def test_grayscale_image_is_converted(self, tmp_path):
        _make_image(tmp_path / "gray.png", grayscale=True)
        arr, was_gray = load_rgb_image(tmp_path / "gray.png")
        assert was_gray is True
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8

        out = tmp_path / "out"
        summary = run_custom_demo(
            str(tmp_path), seg_predictor=DummySegPredictor(),
            depth_predictor=DummyDepthPredictor(), output_dir=str(out),
        )
        assert summary["num_images"] == 1
        assert (out / "gray_original.png").is_file()

    def test_corrupt_image_clear_error(self, tmp_path):
        _make_image(tmp_path / "broken.png", corrupt=True)
        with pytest.raises(CustomDemoError, match="broken.png"):
            load_rgb_image(tmp_path / "broken.png")

    def test_corrupt_image_fails_run_with_file_name(self, tmp_path):
        _make_image(tmp_path / "broken.png", corrupt=True)
        with pytest.raises(CustomDemoError, match="broken.png"):
            run_custom_demo(
                str(tmp_path), seg_predictor=DummySegPredictor(),
                depth_predictor=DummyDepthPredictor(),
                output_dir=str(tmp_path / "out"),
            )

    def test_missing_checkpoint_clear_error(self, tmp_path):
        with pytest.raises(CustomDemoError, match="U-Net checkpoint file not found"):
            validate_checkpoints(str(tmp_path / "missing_unet.pth"),
                                 str(tmp_path / "missing_midas.pt"))
        unet = tmp_path / "unet.pt"
        unet.touch()
        with pytest.raises(CustomDemoError, match="MiDaS weights file not found"):
            validate_checkpoints(str(unet), str(tmp_path / "missing_midas.pt"))


# ----------------------------------------------------------------------- #
#  11. Step 11 / Step 15 behavior preserved
# ----------------------------------------------------------------------- #

class TestRegressionSafety:
    def test_step11_single_image_cli_preserved(self):
        import main as cli

        args = cli.build_parser().parse_args(
            ["--image", "x.png", "--output-dir", "out",
             "--unet-checkpoint", "a.pt", "--midas-weights", "b.pt",
             "--no-visualization", "--device", "cpu"]
        )
        assert args.image == "x.png"
        assert args.output_dir == "out"
        assert args.unet_checkpoint == "a.pt"
        assert args.midas_weights == "b.pt"
        assert args.no_visualization is True
        assert args.device == "cpu"

    def test_step15_evaluate_single_image_still_works(self, tmp_path):
        from evaluation.evaluate_pipeline import evaluate_single_image

        _make_image(tmp_path / "old.png", size=(48, 96))
        result = evaluate_single_image(
            tmp_path / "old.png", DummySegPredictor(), DummyDepthPredictor(),
            {}, device=torch.device("cpu"),
        )
        assert result["image"] == "old.png"
        assert "difficulty" in result
        assert "scene" in result

    def test_build_scene_report_is_json_serializable(self, tmp_path):
        from evaluation.evaluate_pipeline import evaluate_single_image

        _make_image(tmp_path / "s.png", size=(48, 96))
        result = evaluate_single_image(
            tmp_path / "s.png", DummySegPredictor(), DummyDepthPredictor(),
            {}, device=torch.device("cpu"),
        )
        fusion = result["_internals"]["fusion_result"]
        report = build_scene_report(tmp_path / "s.png", result, fusion)
        json.dumps(report)  # must not raise


# ----------------------------------------------------------------------- #
#  Regression: bounded inference resolution (small-GPU CUDA OOM fix)
# ----------------------------------------------------------------------- #

class TestSafeInferenceResolution:
    """U-Net inference must run at a bounded resolution while MiDaS stays at
    the bounded 384x384 input, with predictions resized back to the original
    resolution for final output alignment (padded/CPU only, mocked models)."""

    def test_safe_unet_size_picks_smallest_configured_candidate(self):
        from evaluation.evaluate_pipeline import _safe_unet_inference_size

        cfg = load_config("unet")
        size = _safe_unet_inference_size(cfg)
        assert size == (256, 512)  # data.image_size is the safe budget
        assert _safe_unet_inference_size({}) is None

    def test_load_unet_bounds_inference_resolution(self):
        from evaluation.evaluate_pipeline import _load_unet

        with patch("models.unet.inference.UNet"), patch(
            "models.unet.inference.UNetInference"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            predictor = _load_unet(
                "checkpoints/unet_cityscapes.pth",
                load_config("unet"),
                torch.device("cpu"),
            )
            assert predictor is mock_cls.return_value
            image_size = mock_cls.call_args.kwargs["image_size"]
            assert image_size is not None, "U-Net must not run at source size"
            assert image_size[0] * image_size[1] <= 256 * 512

    def test_load_midas_bounds_input_size(self):
        from evaluation.evaluate_pipeline import _load_midas

        with patch("torch.hub.load", return_value=torch.nn.Module()), patch(
            "models.midas.model._load_weights"
        ), patch("models.midas.inference.MidDepthPredictor") as mock_cls:
            mock_cls.return_value = MagicMock()
            _load_midas(
                "checkpoints/dpt_large_384.pt",
                load_config("midas"),
                torch.device("cpu"),
            )
            kwargs = mock_cls.call_args.kwargs
            assert kwargs["input_size"] == 384
            assert kwargs["use_official_transform"] is False

    def test_original_resolution_preserved_with_bounded_predictors(self, tmp_path):
        """Predictors returning low-res maps (as cap+resize-back does) must
        still yield full-resolution outputs for fusion and reporting."""
        from evaluation.evaluate_pipeline import evaluate_single_image

        class SmallMapSegPredictor:
            model = MagicMock()
            model.eval = MagicMock()

            def predict(self, image, return_confidence=False):
                seg = np.zeros((16, 32), dtype=np.int64)
                seg[4:12, 8:24] = 13
                out = torch.from_numpy(seg)
                if return_confidence:
                    return out, torch.full((16, 32), 0.9, dtype=torch.float32)
                return out

        class SmallMapDepthPredictor:
            model = MagicMock()
            model.eval = MagicMock()

            def predict(self, image):
                h, w = 16, 32
                yy = np.arange(h, dtype=np.float64)[:, None]
                depth = np.broadcast_to(yy / max(h - 1, 1) * 50.0, (h, w))
                return torch.from_numpy(depth.copy())

        _make_image(tmp_path / "big.png", size=(48, 96))
        result = evaluate_single_image(
            tmp_path / "big.png",
            SmallMapSegPredictor(),
            SmallMapDepthPredictor(),
            {},
            device=torch.device("cpu"),
        )
        internals = result["_internals"]
        assert internals["segmentation"].shape == (48, 96)
        assert internals["depth"].shape == (48, 96)
        assert internals["fusion_result"].seg_mask.shape == (48, 96)