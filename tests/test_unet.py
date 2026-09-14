import pytest
import torch

from models.unet.model import UNet
from utils.config import load_config
from utils.device import cuda_available


@pytest.fixture(scope="module")
def cpu():
    return torch.device("cpu")


def test_model_creation(cpu):
    model = UNet(num_classes=19).to(cpu)
    assert isinstance(model, torch.nn.Module)


def test_forward_pass_shape(cpu):
    model = UNet(num_classes=19).to(cpu)
    x = torch.randn(2, 3, 128, 256)
    y = model(x)
    assert tuple(y.shape) == (2, 19, 128, 256)


@pytest.mark.parametrize("num_classes", [5, 32, 1])
def test_different_num_classes(cpu, num_classes):
    model = UNet(num_classes=num_classes).to(cpu)
    y = model(torch.randn(1, 3, 64, 64))
    assert y.shape[1] == num_classes


@pytest.mark.parametrize("size", [(128, 256), (256, 512), (64, 64)])
def test_different_image_sizes_match_output(cpu, size):
    model = UNet(num_classes=19).to(cpu)
    x = torch.randn(1, 3, *size)
    y = model(x)
    assert tuple(y.shape) == (1, 19, *size)


def test_batch_size_greater_than_one(cpu):
    model = UNet(num_classes=19).to(cpu)
    y = model(torch.randn(4, 3, 128, 256))
    assert y.shape[0] == 4


def test_odd_spatial_dimensions_are_aligned(cpu):
    model = UNet(num_classes=19).to(cpu)
    x = torch.randn(2, 3, 100, 150)
    y = model(x)
    assert tuple(y.shape) == (2, 19, 100, 150)


def test_gradient_flow(cpu):
    model = UNet(num_classes=19).to(cpu)
    x = torch.randn(2, 3, 64, 96)
    loss = model(x).mean()
    loss.backward()
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0
    assert all(torch.isfinite(p.grad).all() for p in params_with_grad)


def test_cpu_compatibility(cpu):
    model = UNet(num_classes=8).to(cpu)
    y = model(torch.randn(1, 3, 96, 96))
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
def test_cuda_compatibility_when_available():
    model = UNet(num_classes=19).cuda()
    y = model(torch.randn(2, 3, 64, 64).cuda())
    assert tuple(y.shape) == (2, 19, 64, 64)
    model.cpu()


def test_output_are_raw_logits_not_probabilities(cpu):
    model = UNet(num_classes=19).to(cpu)
    y = model(torch.randn(2, 3, 64, 64))
    assert tuple(y.shape) == (2, 19, 64, 64)
    probabilities = torch.softmax(y, dim=1)
    assert not torch.allclose(y, probabilities)
    assert not torch.allclose(y.sum(dim=1), torch.ones_like(y.sum(dim=1)))


def test_in_channels_is_configurable(cpu):
    model = UNet(num_classes=19, in_channels=1).to(cpu)
    y = model(torch.randn(1, 1, 64, 64))
    assert tuple(y.shape) == (1, 19, 64, 64)


def test_base_channels_is_configurable(cpu):
    model = UNet(num_classes=19, base_channels=8).to(cpu)
    y = model(torch.randn(1, 3, 128, 256))
    assert tuple(y.shape) == (1, 19, 128, 256)


def test_from_config_uses_model_group(cpu):
    cfg = load_config("unet")
    model = UNet.from_config(cfg).to(cpu)
    y = model(torch.randn(1, 3, 128, 256))
    assert tuple(y.shape) == (1, 19, 128, 256)