from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from models.unet.inference import InferenceError, UNetInference, load_model
from models.unet.model import UNet
from utils.config import load_config


def _synthetic_rgb(size=(256, 512)):
    h, w = size
    array = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return array


@pytest.fixture
def small_model():
    return UNet(num_classes=8, base_channels=4)


@pytest.fixture
def inference(small_model):
    return UNetInference(model=small_model, device="cpu")


def test_inference_object_creation(small_model):
    inferencer = UNetInference(model=small_model, device="cpu")
    assert inferencer.model is small_model
    assert not inferencer.model.training


def test_prediction_from_random_model_builds_model():
    inferencer = UNetInference(num_classes=19, device="cpu")
    assert isinstance(inferencer.model, UNet)


def test_single_image_inference_shape(inference):
    image = _synthetic_rgb((256, 512))
    prediction = inference.predict(image)
    assert tuple(prediction.shape) == (256, 512)
    assert prediction.dtype == torch.long


@pytest.mark.parametrize("num_classes", [19, 5])
def test_class_range_within_num_classes(num_classes):
    inferencer = UNetInference(
        model=UNet(num_classes=num_classes, base_channels=4), device="cpu"
    )
    prediction = inferencer.predict(_synthetic_rgb((128, 256)))
    assert prediction.min().item() >= 0
    assert prediction.max().item() <= num_classes - 1


@pytest.mark.parametrize("size", [(128, 256), (256, 512), (100, 150)])
def test_different_image_sizes_return_original_resolution(inference, size):
    prediction = inference.predict(_synthetic_rgb(size))
    assert tuple(prediction.shape) == size


@pytest.mark.parametrize("size", [(128, 256), (256, 512), (100, 150)])
def test_output_resized_back_to_original_resolution(size):
    inferencer = UNetInference(
        model=UNet(num_classes=8, base_channels=4), device="cpu", image_size=(128, 256)
    )
    h, w = size
    prediction = inferencer.predict(_synthetic_rgb(size))
    assert tuple(prediction.shape) == (h, w)


def test_batch_inference(small_model):
    inferencer = UNetInference(model=small_model, device="cpu")
    batch = torch.randn(4, 3, 128, 256)
    predictions = inferencer.predict_batch(batch)
    assert tuple(predictions.shape) == (4, 128, 256)
    assert predictions.dtype == torch.long
    assert predictions.min().item() >= 0
    assert predictions.max().item() <= 7


def test_no_gradient_created(inference):
    x = torch.rand(3, 96, 128, requires_grad=True)
    prediction = inference.predict(x)
    assert prediction.requires_grad is False
    with pytest.raises(RuntimeError):
        prediction.sum().backward()
    assert all(p.grad is None for p in inference.model.parameters())


def test_model_in_eval_mode(inference):
    inference.model.train()
    inference.predict(_synthetic_rgb((64, 96)))
    assert inference.model.training is False


def test_checkpoint_state_dict_format(small_model, tmp_path):
    path = tmp_path / "state.pt"
    torch.save(small_model.state_dict(), path)
    inferencer = UNetInference(
        model=UNet(num_classes=8, base_channels=4), device="cpu", checkpoint=path
    )
    prediction = inferencer.predict(_synthetic_rgb((64, 96)))
    assert tuple(prediction.shape) == (64, 96)


def test_checkpoint_dictionary_format(small_model, tmp_path):
    path = tmp_path / "dict.pt"
    torch.save({"model_state_dict": small_model.state_dict()}, path)
    inferencer = UNetInference(
        model=UNet(num_classes=8, base_channels=4), device="cpu", checkpoint=path
    )
    assert tuple(inferencer.predict(_synthetic_rgb((64, 96))).shape) == (64, 96)


def test_load_model_module_function(small_model, tmp_path):
    path = tmp_path / "state.pt"
    torch.save(small_model.state_dict(), path)
    model = load_model(UNet(num_classes=8, base_channels=4), weight_path=path, device="cpu")
    assert model.training is False


def test_missing_checkpoint_file_raises(inference, tmp_path):
    with pytest.raises(InferenceError, match="not found"):
        UNetInference(model=inference.model, checkpoint=tmp_path / "nope.pt")


def test_corrupted_checkpoint_raises(inference, tmp_path):
    path = tmp_path / "corrupt.pt"
    Path(path).write_bytes(b"not a torch checkpoint")
    with pytest.raises(InferenceError):
        UNetInference(model=inference.model, checkpoint=path)


def test_incompatible_num_classes_checkpoint_raises(tmp_path):
    checkpoint_model = UNet(num_classes=19, base_channels=4)
    path = tmp_path / "c19.pt"
    torch.save(checkpoint_model.state_dict(), path)
    with pytest.raises(InferenceError, match="incompatible"):
        UNetInference(
            model=UNet(num_classes=5, base_channels=4), device="cpu", checkpoint=path
        )


def test_incompatible_architecture_checkpoint_raises(tmp_path):
    checkpoint_model = UNet(num_classes=8, base_channels=8)
    path = tmp_path / "arch.pt"
    torch.save(checkpoint_model.state_dict(), path)
    with pytest.raises(InferenceError, match="incompatible"):
        UNetInference(
            model=UNet(num_classes=8, base_channels=4), device="cpu", checkpoint=path
        )


def test_nearest_neighbour_output_resize():
    inferencer = UNetInference(
        model=UNet(num_classes=8, base_channels=4), device="cpu", image_size=(128, 256)
    )
    prediction = inferencer.predict(_synthetic_rgb((100, 150)))
    assert tuple(prediction.shape) == (100, 150)
    assert prediction.dtype == torch.long
    assert set(prediction.unique().tolist()).issubset(set(range(8)))


def test_raw_integer_prediction_not_probabilities(inference):
    prediction = inference.predict(_synthetic_rgb((64, 96)), return_confidence=False)
    assert prediction.dtype == torch.long
    assert prediction.min().item() >= 0
    assert prediction.max().item() <= 7


def test_confidence_output_is_probability(inference):
    prediction, confidence = inference.predict(
        _synthetic_rgb((100, 150)), return_confidence=True
    )
    assert tuple(prediction.shape) == (100, 150)
    assert tuple(confidence.shape) == (100, 150)
    assert confidence.dtype == torch.float32
    assert confidence.min().item() >= 0.0
    assert confidence.max().item() <= 1.0


def test_rgb_pil_image_input(inference):
    pil = Image.fromarray(_synthetic_rgb((128, 256)), mode="RGB")
    assert tuple(inference.predict(pil).shape) == (128, 256)


def test_float_tensor_input(inference):
    tensor = torch.rand(3, 128, 256)
    assert tuple(inference.predict(tensor).shape) == (128, 256)


def test_unsupported_input_type_raises(inference):
    with pytest.raises(InferenceError, match="Unsupported image type"):
        inference.predict([1, 2, 3])


def test_preprocess_returns_batched_normalized_tensor(inference):
    image = _synthetic_rgb((128, 256))
    batch = inference.preprocess(image)
    assert tuple(batch.shape) == (1, 3, 128, 256)
    assert torch.isfinite(batch).all()


def test_predict_batch_requires_4d_tensor(inference):
    with pytest.raises(InferenceError, match="B, 3, H, W"):
        inference.predict_batch(torch.randn(3, 128, 256))


def test_cuda_request_raises_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA available; build without GPU to test the error path")
    with pytest.raises(InferenceError, match="CUDA requested but not available"):
        UNetInference(num_classes=8, device="cuda")


def test_from_config_builds_inference():
    cfg = load_config("unet", overrides={"inference": {"image_size": None}})
    inferencer = UNetInference.from_config(cfg, model=UNet(num_classes=19, base_channels=4))
    prediction = inferencer.predict(_synthetic_rgb((96, 128)))
    assert tuple(prediction.shape) == (96, 128)
    assert prediction.max().item() <= 18