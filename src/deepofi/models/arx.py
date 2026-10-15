"""ARX: linear regression of the H forward returns on the last W events of the input (W x F lags).

The paper fits it like the networks (Adam, lr 1e-4, early stopping). Here `fit` solves ridge
regression in closed form instead, accumulating X'X and X'Y over batches and choosing the penalty
on the validation set (DEVIATIONS.md). Training through engine.train still works for comparison.
"""
import torch
from torch import nn

MODEL_PARAMS = {"window": 100}
TRAINING_PARAMS = {
    "learning_rate": 1e-4,
    # Penalty grid relative to the average eigenvalue of the centered X'X.
    "ridge_grid": [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0],
}


class ARX(nn.Module):
    def __init__(self, n_features, n_horizons, window=100):
        super().__init__()
        self.linear = nn.Linear(window * n_features, n_horizons)

    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))


def build_model(n_features, n_horizons, window=100):
    return ARX(n_features, n_horizons, window)


class _Moments:
    """Running sums for linear regression in float64: n, sum x, sum y, X'X, X'Y, sum y^2."""

    def __init__(self):
        self.n = 0

    def update(self, x, y):
        x = x.flatten(start_dim=1).double()
        y = y.double()
        if self.n == 0:
            self.sx = torch.zeros(x.shape[1], dtype=torch.float64)
            self.sy = torch.zeros(y.shape[1], dtype=torch.float64)
            self.sxx = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
            self.sxy = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64)
            self.syy = torch.zeros(y.shape[1], dtype=torch.float64)
        self.n += x.shape[0]
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.sxx += x.T @ x
        self.sxy += x.T @ y
        self.syy += (y * y).sum(0)

    @classmethod
    def from_loader(cls, loader):
        moments = cls()
        for x, y in loader:
            moments.update(x, y)
        return moments

    def mse(self, beta, bias):
        """Per-output mean squared error of x @ beta + bias, from the sums alone."""
        sse = (
            self.syy
            - 2 * (beta * self.sxy).sum(0)
            - 2 * bias * self.sy
            + (beta * (self.sxx @ beta)).sum(0)
            + 2 * bias * (self.sx @ beta)
            + self.n * bias**2
        )
        return sse / self.n


def fit(model, train_loader, val_loader, ridge_grid=TRAINING_PARAMS["ridge_grid"], **_):
    """Closed-form ridge with an unpenalized intercept; the penalty minimizing mean val MSE wins."""
    train = _Moments.from_loader(train_loader)
    val = _Moments.from_loader(val_loader)

    mean_x, mean_y = train.sx / train.n, train.sy / train.n
    cxx = train.sxx - train.n * torch.outer(mean_x, mean_x)
    cxy = train.sxy - train.n * torch.outer(mean_x, mean_y)
    eigenvalues, eigenvectors = torch.linalg.eigh(cxx)
    projected = eigenvectors.T @ cxy
    scale = eigenvalues.clamp(min=0).mean()

    history = []
    best = None
    for alpha in ridge_grid:
        beta = eigenvectors @ (projected / (eigenvalues + alpha * scale)[:, None])
        bias = mean_y - mean_x @ beta
        val_mse = val.mse(beta, bias).mean().item()
        history.append({"alpha": alpha, "train_mse": train.mse(beta, bias).mean().item(), "val_mse": val_mse})
        print(f"ridge alpha {alpha:g} | Train MSE: {history[-1]['train_mse']:.6f} | Val MSE: {val_mse:.6f}")
        if best is None or val_mse < best[0]:
            best = (val_mse, alpha, beta, bias)

    _, alpha, beta, bias = best
    with torch.no_grad():
        model.linear.weight.copy_(beta.T.float())
        model.linear.bias.copy_(bias.float())
    print(f"Selected ridge alpha {alpha:g} on {train.n:,} train / {val.n:,} val windows")
    model.eval()
    return model, history

