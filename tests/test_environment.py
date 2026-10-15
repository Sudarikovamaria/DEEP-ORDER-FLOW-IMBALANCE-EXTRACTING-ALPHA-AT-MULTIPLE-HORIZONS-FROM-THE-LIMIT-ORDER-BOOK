import torch

from deepofi.engine import get_device


def test_get_device_auto_returns_available_backend():
    device = get_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    torch.zeros(2, device=device).sum().item()


def test_get_device_explicit():
    assert get_device("cpu") == torch.device("cpu")
