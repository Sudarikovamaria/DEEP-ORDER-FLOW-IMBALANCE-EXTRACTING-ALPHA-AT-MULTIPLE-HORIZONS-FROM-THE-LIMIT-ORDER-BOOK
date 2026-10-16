"""LSTM-MLP (Section 2.2.3 of the paper): the last LSTM state goes through a ReLU MLP instead of a linear head.

Table 2 gives 3 layers and 8.4e4 (OF) / 9.4e4 (LOB) parameters, truncated to two digits. Their
difference, 4 x hidden x 20 extra inputs = 1.0e4, puts the LSTM at ~125 units. With 128 units only a
middle layer of 56-60 units lands in both [84k, 85k) and [94k, 95k); 60 gives 84,638 / 94,878
(Keras count with one LSTM bias). The LSTM is initialised like lstm.py (Keras defaults, forget bias 1).
"""
from torch import nn

from .lstm import LSTMModel

MODEL_PARAMS = {"hidden_size": 128, "mlp_size": 60}
TRAINING_PARAMS = {"learning_rate": 1e-5, "epochs": 50, "patience": 5}


class LSTMMLP(LSTMModel):
    def __init__(self, n_features, n_horizons, hidden_size=128, mlp_size=60):
        super().__init__(n_features, n_horizons, hidden_size=hidden_size, num_layers=1)
        self.head = nn.Sequential(nn.Linear(hidden_size, mlp_size), nn.ReLU(), nn.Linear(mlp_size, n_horizons))
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


def build_model(n_features, n_horizons, **params):
    return LSTMMLP(n_features, n_horizons, **params)
