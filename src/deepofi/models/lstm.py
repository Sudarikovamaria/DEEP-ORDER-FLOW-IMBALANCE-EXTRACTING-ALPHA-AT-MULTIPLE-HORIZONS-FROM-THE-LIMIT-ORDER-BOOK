import torch
from torch import nn

MODEL_PARAMS = {"hidden_size": 150, "num_layers": 1}
TRAINING_PARAMS = {"learning_rate": 1e-5, "epochs": 50, "patience": 5}


class LSTMModel(nn.Module):
    def __init__(self, n_features, n_horizons, hidden_size=150, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_horizons)

        for layer in range(num_layers):
            weight_ih = getattr(self.lstm, f"weight_ih_l{layer}")
            weight_hh = getattr(self.lstm, f"weight_hh_l{layer}")
            bias_ih = getattr(self.lstm, f"bias_ih_l{layer}")
            bias_hh = getattr(self.lstm, f"bias_hh_l{layer}")

            nn.init.xavier_uniform_(weight_ih)
            nn.init.orthogonal_(weight_hh)
            nn.init.zeros_(bias_ih)
            nn.init.zeros_(bias_hh)

            with torch.no_grad():
                bias_ih[hidden_size:2 * hidden_size].fill_(1.0)
            bias_hh.requires_grad_(False)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


def build_model(n_features, n_horizons, **params):
    return LSTMModel(n_features, n_horizons, **params)