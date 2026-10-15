"""CNN-LSTM (Table 1 of the paper, after DeepLOB by Zhang, Zohren and Roberts, 2019).

The input [B, W, F] is treated as a one-channel W x F image (time x features):
    block 1  Conv 1x2 stride 2 (price with volume), then two Conv 4x1      LOB only: 40 -> 20
    block 2  Conv 1x2 stride 2 (bid with ask of a level), two Conv 4x1     20 -> 10
    block 3  Conv 1x10 (all levels), two Conv 4x1                          10 -> 1
    block 4  inception: 1x1 -> 3x1 | 1x1 -> 5x1 | max-pool 3x1 -> 1x1
    block 5  LSTM over time, last hidden state -> linear head
LOB (F=40) runs blocks 1-5, OF (F=20) blocks 2-5 and OFI (F=10) blocks 3-5. All activations are
Leaky ReLU; time convolutions use "same" padding, so the window keeps its W events.
"""
import torch
from torch import nn

MODEL_PARAMS = {"conv_filters": 32, "inception_filters": 64, "lstm_units": 64, "negative_slope": 0.01}
TRAINING_PARAMS = {"learning_rate": 1e-3, "epochs": 50, "patience": 5}

FIRST_BLOCK = {40: 1, 20: 2, 10: 3}


def conv_block(in_channels, filters, kernel, stride, negative_slope):
    layers = [nn.Conv2d(in_channels, filters, kernel, stride=stride), nn.LeakyReLU(negative_slope)]
    for _ in range(2):
        # "same" padding for an even kernel, laid out as in Keras: 1 step before, 2 after.
        layers += [nn.ZeroPad2d((0, 0, 1, 2)), nn.Conv2d(filters, filters, (4, 1)), nn.LeakyReLU(negative_slope)]
    return nn.Sequential(*layers)


class Inception(nn.Module):
    def __init__(self, in_channels, filters, negative_slope):
        super().__init__()
        act = lambda: nn.LeakyReLU(negative_slope)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, filters, 1), act(), nn.Conv2d(filters, filters, (3, 1), padding="same"), act()
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, filters, 1), act(), nn.Conv2d(filters, filters, (5, 1), padding="same"), act()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), nn.Conv2d(in_channels, filters, 1), act()
        )

    def forward(self, x):
        return torch.cat([self.branch3(x), self.branch5(x), self.pool(x)], dim=1)


class CNNLSTM(nn.Module):
    def __init__(self, n_features, n_horizons, conv_filters=32, inception_filters=64, lstm_units=64, negative_slope=0.01):
        super().__init__()
        if n_features not in FIRST_BLOCK:
            raise ValueError(f"CNN-LSTM expects 40 (LOB), 20 (OF) or 10 (OFI) features, got {n_features}")
        first = FIRST_BLOCK[n_features]
        self.levels = 10
        self.interleave = n_features == 20

        blocks = []
        channels = 1
        if first <= 1:
            blocks.append(conv_block(channels, conv_filters, (1, 2), (1, 2), negative_slope))
            channels = conv_filters
        if first <= 2:
            blocks.append(conv_block(channels, conv_filters, (1, 2), (1, 2), negative_slope))
            channels = conv_filters
        blocks.append(conv_block(channels, conv_filters, (1, self.levels), (1, 1), negative_slope))
        self.convolutions = nn.Sequential(*blocks)

        self.inception = Inception(conv_filters, inception_filters, negative_slope)
        self.lstm = nn.LSTM(3 * inception_filters, lstm_units, batch_first=True)
        self.head = nn.Linear(lstm_units, n_horizons)

        # Forget-gate bias 1 (gate order in PyTorch: input, forget, cell, output).
        with torch.no_grad():
            self.lstm.bias_ih_l0[lstm_units:2 * lstm_units].fill_(1.0)
            self.lstm.bias_hh_l0[lstm_units:2 * lstm_units].fill_(0.0)

    def forward(self, x):
        if self.interleave:
            # [bOF_1..bOF_L, aOF_1..aOF_L] -> [bOF_1, aOF_1, bOF_2, aOF_2, ...] so block 2 pairs a level's sides.
            x = torch.stack([x[..., :self.levels], x[..., self.levels:]], dim=-1).flatten(start_dim=-2)
        x = self.convolutions(x.unsqueeze(1))   # [B, C, W, 1]
        x = self.inception(x)                    # [B, 3 * inception_filters, W, 1]
        x = x.squeeze(-1).transpose(1, 2)        # [B, W, 3 * inception_filters]
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


def build_model(n_features, n_horizons, **params):
    return CNNLSTM(n_features, n_horizons, **params)
