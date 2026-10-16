"""MLP (Section 2.2.2 of the paper): the W x F window flattened into one vector, ReLU hidden layers.

Table 2 gives 4 layers and ~1.3e6 (OF) / ~2.3e6 (LOB) parameters but not the widths. Their
difference, 2000 extra inputs x first-layer width = 1.0e6, fixes the first layer at ~500; 512-512-128
gives 1.35e6 / 2.38e6 parameters, which Table 2 truncates to 1.3e6 / 2.3e6.
"""
from torch import nn

MODEL_PARAMS = {"window": 100, "hidden_sizes": (512, 512, 128)}
TRAINING_PARAMS = {"learning_rate": 1e-5, "epochs": 50, "patience": 5}


class MLP(nn.Module):
    def __init__(self, n_features, n_horizons, window=100, hidden_sizes=(512, 512, 128)):
        super().__init__()
        layers = [nn.Flatten()]
        width = window * n_features
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, n_horizons))
        self.network = nn.Sequential(*layers)

        # Keras defaults, as in the paper: Glorot-uniform weights, zero biases.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


def build_model(n_features, n_horizons, **params):
    return MLP(n_features, n_horizons, **params)
