import pytest
import torch

from deepofi.models.cnn_lstm import CNNLSTM


@pytest.mark.parametrize("n_features, n_blocks", [(40, 3), (20, 2), (10, 1)])
def test_blocks_depend_on_input(n_features, n_blocks):
    model = CNNLSTM(n_features, 10)
    assert len(model.convolutions) == n_blocks
    assert model(torch.randn(3, 100, n_features)).shape == (3, 10)


def test_of_is_interleaved_by_level_before_block_two():
    model = CNNLSTM(20, 10)
    seen = {}

    def keep_input(module, inputs, output):
        seen["x"] = inputs[0]

    model.convolutions.register_forward_hook(keep_input)
    x = torch.arange(20.0).repeat(1, 100, 1)  # bOF_i = i - 1, aOF_i = 10 + i - 1
    model(x)
    assert seen["x"][0, 0, 0, :6].tolist() == [0, 10, 1, 11, 2, 12]


def test_forget_gate_bias_and_window_length():
    model = CNNLSTM(20, 10, lstm_units=64)
    assert torch.all(model.lstm.bias_ih_l0[64:128] == 1) and torch.all(model.lstm.bias_hh_l0[64:128] == 0)
    features = model.inception(model.convolutions(torch.randn(2, 1, 100, 20)))
    assert features.shape == (2, 192, 100, 1)


def test_rejects_unknown_width():
    with pytest.raises(ValueError):
        CNNLSTM(30, 10)
