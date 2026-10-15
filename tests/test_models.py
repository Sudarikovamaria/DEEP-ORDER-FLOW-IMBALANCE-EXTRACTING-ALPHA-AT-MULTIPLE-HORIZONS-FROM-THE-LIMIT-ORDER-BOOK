"""Interface check for every registered model; modules that are still empty are skipped."""
import importlib

import pytest
import torch

from deepofi.models import MODELS, get_model


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("n_features", [20, 40, 10])
def test_model_interface(model_name, n_features):
    module = importlib.import_module(f"deepofi.models.{MODELS[model_name]}")
    if not hasattr(module, "build_model"):
        pytest.skip(f"{model_name} is not implemented yet")

    model, params = get_model(model_name, n_features=n_features, n_horizons=10)
    assert "learning_rate" in params

    x = torch.randn(4, 100, n_features)
    y = model(x)
    assert y.shape == (4, 10)
    y.sum().backward()
