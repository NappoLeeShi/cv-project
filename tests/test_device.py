import torch
import pytest

from utils.device import cuda_available, device_summary, resolve_device


def test_resolve_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto():
    device = resolve_device("auto")
    expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert device == expected


def test_resolve_none_means_auto():
    assert resolve_device(None) == resolve_device("auto")


def test_resolve_unknown_device_raises():
    with pytest.raises((ValueError, RuntimeError)):
        resolve_device("not_a_device")


def test_cuda_available_flag():
    assert isinstance(cuda_available(), bool)


def test_device_summary_contains_fields():
    summary = device_summary()
    assert isinstance(summary, dict)
    assert "device" in summary
    assert "cuda_available" in summary
    assert "torch_version" in summary