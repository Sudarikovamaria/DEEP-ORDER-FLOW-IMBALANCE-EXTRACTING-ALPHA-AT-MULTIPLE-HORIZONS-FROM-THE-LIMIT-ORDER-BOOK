import torch
from torch.utils.data import DataLoader, TensorDataset

from deepofi.models.arx import _Moments, build_model, fit


def linear_data(n, window=5, n_features=3, n_horizons=2, noise=0.01, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, window, n_features, generator=generator)
    beta = torch.randn(window * n_features, n_horizons, generator=torch.Generator().manual_seed(42))
    y = x.flatten(1) @ beta + 0.5 + noise * torch.randn(n, n_horizons, generator=generator)
    return x, y, beta


def test_moments_mse_matches_direct_computation():
    x, y, _ = linear_data(500)
    moments = _Moments.from_loader(DataLoader(TensorDataset(x, y), batch_size=64))
    beta = torch.randn(15, 2, dtype=torch.float64)
    bias = torch.tensor([0.1, -0.2], dtype=torch.float64)

    direct = ((y.double() - x.flatten(1).double() @ beta - bias) ** 2).mean(0)
    torch.testing.assert_close(moments.mse(beta, bias), direct)


def test_ridge_recovers_coefficients_and_intercept():
    x, y, beta = linear_data(4000)
    x_val, y_val, _ = linear_data(1000, seed=1)
    model = build_model(n_features=3, n_horizons=2, window=5)

    model, history = fit(
        model,
        DataLoader(TensorDataset(x, y), batch_size=256),
        DataLoader(TensorDataset(x_val, y_val), batch_size=256),
        ridge_grid=[1e-6, 1e-2, 10.0],
    )

    assert min(history, key=lambda h: h["val_mse"])["alpha"] == 1e-6
    torch.testing.assert_close(model.linear.weight.T, beta, atol=5e-3, rtol=0)
    torch.testing.assert_close(model.linear.bias, torch.tensor([0.5, 0.5]), atol=5e-3, rtol=0)
    assert model(x[:4]).shape == (4, 2)


def test_merged_moments_equal_moments_of_concatenated_data():
    x1, y1, _ = linear_data(300, seed=2)
    x2, y2, _ = linear_data(200, seed=3)
    merged = _Moments.from_loader(DataLoader(TensorDataset(x1, y1), batch_size=64)).merge(
        _Moments.from_loader(DataLoader(TensorDataset(x2, y2), batch_size=64)))
    joint = _Moments.from_loader(DataLoader(TensorDataset(torch.cat([x1, x2]), torch.cat([y1, y2])), batch_size=64))
    assert merged.n == joint.n == 500
    for name in ("sx", "sy", "sxx", "sxy", "syy"):
        torch.testing.assert_close(getattr(merged, name), getattr(joint, name))
