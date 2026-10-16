"""Trainable parameter counts against Table 2 of the paper (Keras counts: one LSTM bias per gate)."""
import math

import pytest
import torch

from deepofi.models import get_model

TABLE_2 = {  # model: (OF, LOB)
    "cnn_lstm": (1.3e5, 1.4e5),
    "lstm": (1.0e5, 1.1e5),
    "lstm3": (4.6e5, 4.7e5),
    "lstm_mlp": (8.4e4, 9.4e4),
    "mlp": (1.3e6, 2.3e6),
}


@pytest.mark.parametrize("model_name", TABLE_2)
@pytest.mark.parametrize("input_index, n_features", [(0, 20), (1, 40)])
def test_parameter_counts_match_table_2(model_name, input_index, n_features):
    model, _ = get_model(model_name, n_features=n_features)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = TABLE_2[model_name][input_index]
    # Table 2 truncates to two significant digits: 1.1e5 means [110,000, 120,000).
    step = 10 ** (math.floor(math.log10(expected)) - 1)
    assert expected <= trainable < expected + step, f"{model_name}: {trainable:,} vs {expected:,.0f}"


@pytest.mark.parametrize("model_name", ["mlp", "lstm_mlp"])
def test_new_models_learn_a_linear_signal(model_name):
    torch.manual_seed(0)
    model, _ = get_model(model_name, n_features=10)
    x = torch.randn(512, 100, 10)
    y = x[:, -5:, 0].sum(dim=1, keepdim=True).repeat(1, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first
