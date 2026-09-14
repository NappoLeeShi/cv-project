import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from models.midas.inference import MidDepthPredictor
from models.midas.model import (
    DEFAULT_VARIANT,
    MiDaSModel,
    MiDaSError,
    SUPPORTED_VARIANTS,
    build_midas_model,
)
from utils.config import load_config


class _StubDepthBackend(nn.Module):
    """Tiny backend that mimics MiDaS: returns a single-channel inverse-depth
    map at the model input resolution (no download, fully on CPU)."""

    def forward(self, x):
        b, _, _, w = x.shape
        base = x.float().mean(dim=1, keepdim=True)
        ramp = torch.linspace(0.0, 0.5, steps=w, device=x.device).view(1, 1, 1, w)
        return (base + ramp)[:, 0]


@pytest.fixture
def stub_model():
    return MiDaSModel(backend=_StubDepthBackend(), variant="dpt_hybrid")


@pytest.fixture
def predictor(stub_model):
    return MidDepthPredictor(model=stub_model, device="cpu")


def _synthetic_rgb(size=(128, 256)):
    h, w = size
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _pil(size=(128, 256)):
    return Image.fromarray(_synthetic_rgb(size), mode="RGB")


# Test 1 -- Configuration
def test_configuration_loads_and_exposes_midas_settings():
    cfg = load_config("midas")
    assert cfg["model"]["name"] == "midas"
    assert cfg["model"]["variant"] in SUPPORTED_VARIANTS
    assert cfg["model"]["source"] == "hub"
    assert cfg["model"]["weights_path"] is None
    assert cfg["preprocessing"]["input_size"] == 384
    assert cfg["preprocessing"]["use_official_transform"] is True
    assert cfg["inference"]["device"] == "auto"
    assert cfg["inference"]["clip_percentile"] == [0.05, 0.995]
    assert cfg["output"]["larger_is_farther"] is True
    assert cfg["depth"]["representation"] == "relative"


# Test 2 -- Model wrapper creation (no download)
def test_model_wrapper_creation_without_download(stub_model):
    assert isinstance(stub_model, MiDaSModel)
    assert stub_model.name == "midas"
    assert stub_model.variant == "dpt_hybrid"


def test_wrapper_describes_model_contract():
    model = MiDaSModel(backend=_StubDepthBackend())
    info = model.describe()
    assert info["name"] == "midas"
    assert info["representation"] == "inverse relative depth"
    assert info["is_metric"] is False
    assert info["metric_scale"] is None


def test_unsupported_variant_rejected_without_network():
    with pytest.raises(MiDaSError, match="variant"):
        MiDaSModel(backend=_StubDepthBackend(), variant="dpt_tiny")
    with pytest.raises(MiDaSError, match="variant"):
        build_midas_model(variant="not_a_variant")


def test_wrapper_rejects_non_module_backend():
    with pytest.raises(MiDaSError, match="backend"):
        MiDaSModel(backend="not a module")


# Test 3 -- Input preprocessing
def test_preprocess_pil_image(predictor):
    image = _pil((128, 256))
    batch, source = predictor.preprocess(image)
    assert tuple(batch.shape) == (1, 3, 128, 256)
    assert source == (256, 128)
    assert batch.dtype == torch.float32
    assert batch.is_floating_point()


def test_preprocess_numpy_hwc_image(predictor):
    image = _synthetic_rgb((64, 128))
    batch, source = predictor.preprocess(image)
    assert tuple(batch.shape) == (1, 3, 64, 128)
    assert source == (128, 64)


@pytest.mark.parametrize("size", [(128, 256), (100, 150)])
def test_preprocess_pil_and_numpy_agree(predictor, size):
    array = _synthetic_rgb(size)
    batch_a, _ = predictor.preprocess(Image.fromarray(array))
    batch_b, _ = predictor.preprocess(array)
    assert torch.allclose(batch_a, batch_b, atol=1e-6)


# Test 4 -- Single-image inference
def test_single_image_inference_shape(predictor):
    depth = predictor.predict(_synthetic_rgb((128, 256)))
    assert tuple(depth.shape) == (128, 256)


# Test 5 -- Different image sizes
@pytest.mark.parametrize("size", [(128, 256), (256, 512), (100, 150)])
def test_different_image_sizes_return_original_resolution(predictor, size):
    depth = predictor.predict(_synthetic_rgb(size))
    assert tuple(depth.shape) == size


# Test 6 -- Batch inference
def test_batch_inference_shape(predictor):
    image = _pil((128, 256))
    batch, _ = predictor.preprocess(image)
    stacked = torch.cat([batch, batch], dim=0)
    depth = predictor.predict_batch(stacked)
    assert tuple(depth.shape) == (2, 128, 256)
    single = predictor.predict(image)
    assert torch.allclose(depth[0], single, atol=1e-6)


def test_batch_inference_rejects_non_tensor():
    with pytest.raises(MiDaSError, match=r"\[B, 3, H, W\]"):
        predictor = MidDepthPredictor(model=_StubDepthBackend(), device="cpu")
        predictor.predict_batch(np.zeros((1, 3, 64, 64), dtype=np.float32))


# Test 7 -- Floating-point output
def test_output_is_floating_point(predictor):
    depth = predictor.predict(_synthetic_rgb((128, 256)))
    assert depth.dtype == torch.float32
    assert depth.is_floating_point()


# Test 8 -- Relative depth representation
def test_relative_depth_is_promoted_not_metric(predictor, stub_model):
    assert predictor.depth_representation == "relative"
    assert predictor.metric_scale is None
    assert predictor.raw_output_larger_is_closer is True
    assert stub_model.is_metric is False
    assert MiDaSModel.raw_output_is_inverse_relative is True


# Test 9 -- No gradient accumulation
def test_inference_produces_no_gradients(predictor):
    depth = predictor.predict(_synthetic_rgb((128, 256)))
    assert depth.requires_grad is False
    for param in predictor.model.parameters():
        assert param.grad is None or param.grad.abs().sum().item() == 0.0
    with pytest.raises(RuntimeError):
        depth.sum().backward()


def test_input_tensors_do_not_require_grad(predictor):
    image = torch.rand(3, 64, 128)
    with torch.no_grad():
        depth = predictor.predict(image)
    assert depth.requires_grad is False


# Test 10 -- Evaluation mode
def test_model_placed_in_evaluation_mode(predictor):
    depth = predictor.predict(_synthetic_rgb((64, 128)))
    assert not predictor.model.training
    assert not predictor.model.backend.training
    predictor.model.train()
    predictor.predict(_synthetic_rgb((64, 128)))
    assert not predictor.model.training


# Test 11 -- CPU compatibility
def test_cpu_inference_works():
    predictor = MidDepthPredictor(model=_StubDepthBackend(), device="cpu")
    depth = predictor.predict(_synthetic_rgb((80, 160)))
    assert tuple(depth.shape) == (80, 160)


# Test 12 -- Continuous output resizing back to the source resolution
def test_output_resized_back_with_bilinear_interpolation(predictor):
    size = (100, 150)
    image = _pil(size)
    depth = predictor.predict(image)
    assert tuple(depth.shape) == size
    batch, _ = predictor.preprocess(image)
    raw = predictor.model(batch)
    expected = F.interpolate(
        raw.unsqueeze(1), size=(100, 150), mode="bilinear", align_corners=False
    ).squeeze(1)
    assert torch.allclose(depth, expected[0], atol=1e-5)


@pytest.mark.parametrize("input_size", [384, 64])
def test_internal_resolution_resized_back_to_source(predictor, input_size):
    predictor = MidDepthPredictor(
        model=_StubDepthBackend(), device="cpu", input_size=input_size
    )
    depth = predictor.predict(_synthetic_rgb((96, 128)))
    assert tuple(depth.shape) == (96, 128)


# Test 13 -- No arbitrary meter conversion
def test_no_meter_conversion(predictor):
    depth = predictor.predict(np.zeros((64, 128, 3), dtype=np.uint8))
    assert predictor.metric_scale is None
    assert (depth.abs() < 10.0).all()
    again = predictor.predict(np.zeros((64, 128, 3), dtype=np.uint8))
    assert torch.equal(depth, again)


# Test 14 -- Invalid input
@pytest.mark.parametrize(
    "bad_input",
    [
        [0, 1, 2],
        np.zeros((240, 320), dtype=np.uint8),
        np.zeros((240, 240, 4), dtype=np.uint8),
        torch.zeros(1, 3, 20, 30),
        torch.zeros(1, 1, 20, 30),
    ],
)
def test_invalid_input_raises_clear_error(predictor, bad_input):
    with pytest.raises(MiDaSError):
        predictor.predict(bad_input)


# Official-transform path (offline stub transform)
def test_official_transform_receives_hwc_rgb_and_forwards_output(predictor):
    captured = {}

    def fake_official_transform(array):
        captured["shape"] = array.shape
        captured["dtype"] = array.dtype
        batch = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)
        return batch.unsqueeze(0)

    p = MidDepthPredictor(
        model=_StubDepthBackend(),
        device="cpu",
        transform=fake_official_transform,
        use_official_transform=True,
    )
    assert p.use_official_transform is True
    depth = p.predict(_synthetic_rgb((64, 128)))
    assert tuple(depth.shape) == (64, 128)
    assert captured["shape"] == (64, 128, 3)
    assert captured["dtype"] == np.uint8


# from_config with an offline model
def test_from_config_builds_offline_predictor():
    cfg = load_config("midas")
    predictor = MidDepthPredictor.from_config(cfg, model=_StubDepthBackend())
    assert predictor.input_size == 384
    depth = predictor.predict(_synthetic_rgb((80, 160)))
    assert tuple(depth.shape) == (80, 160)


# Visualization normalization helper stays separate from raw prediction
def test_visualization_normalization_is_separate_and_inverts_depth(predictor):
    depth = predictor.predict(np.zeros((32, 64, 3), dtype=np.uint8))
    viz = predictor.normalize_visualization(depth)
    assert viz.min().item() >= 0.0
    assert viz.max().item() <= 1.0
    assert not torch.equal(depth, viz)


def test_default_variant_is_supported():
    assert DEFAULT_VARIANT in SUPPORTED_VARIANTS