import numpy as np
import torch
from torch import nn

from deepofi.data_process.preprocessing import Normalizer
from deepofi.data_process.synthetic import generate
from deepofi.data_process.windows import WindowDataset, list_dates, make_loaders
from deepofi.engine import train
from deepofi.evaluation.metrics import calculate_metrics, predict


def test_windows_stay_inside_a_day():
    x = [np.arange(10, dtype=np.float32)[:, None], 100 + np.arange(10, dtype=np.float32)[:, None]]
    y = [np.zeros((10, 2), dtype=np.float32)] * 2
    mask = [np.ones((10, 2), dtype=bool), np.ones((10, 2), dtype=bool)]
    mask[1][-2:, 1] = False
    dataset = WindowDataset(list(zip(x, y, mask)), window=4)

    assert len(dataset) == 7 + 5
    first, _ = dataset[0]
    second_day_first, _ = dataset[7]
    assert first[:, 0].tolist() == [0, 1, 2, 3]
    assert second_day_first[:, 0].tolist() == [100, 101, 102, 103]


def test_normalizer_uses_only_fitted_data():
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    normalizer = Normalizer(quantiles=(0.0, 1.0)).fit([train])
    assert np.allclose(normalizer.transform(train).mean(), 0, atol=1e-6)
    assert np.allclose(normalizer.transform(np.array([[100.0]])), (3 - 1.5) / train.std())
    assert np.allclose(normalizer.inverse(normalizer.transform(train)), train, atol=1e-6)


def test_linear_model_learns_synthetic_signal(tmp_path):
    generate(root=tmp_path, tickers=("SYN1",), n_days=4, n_events=20_000)
    dates = list_dates(tmp_path, "SYN1")
    (train_loader, val_loader, test_loader), y_normalizer = make_loaders(
        tmp_path, "SYN1", dates[1:3], dates[:1], dates[3:], window=100, max_samples_per_epoch=40_000
    )
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(100 * 20, 10))

    model, _ = train(model, train_loader, val_loader, learning_rate=1e-3, weights_path="unused.pt", epochs=5, device="cpu", save_weights=False)
    y_true, y_pred = predict(model, test_loader, device="cpu")
    r2 = calculate_metrics(y_true, torch.from_numpy(y_normalizer.inverse(y_pred.double().numpy())))["r2_os"]

    assert np.mean(r2) > 0.05
