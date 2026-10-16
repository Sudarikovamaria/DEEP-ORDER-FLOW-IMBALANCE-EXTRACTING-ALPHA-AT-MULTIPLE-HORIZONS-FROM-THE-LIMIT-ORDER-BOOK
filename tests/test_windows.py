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


def random_days(n_days=3, n_events=300, n_features=5, n_horizons=4, seed=0):
    rng = np.random.default_rng(seed)
    days = []
    for _ in range(n_days):
        mask = rng.random((n_events, n_horizons)) > 0.05
        days.append((rng.normal(size=(n_events, n_features)).astype(np.float32), rng.normal(size=(n_events, n_horizons)).astype(np.float32), mask))
    return days


def test_batched_windows_equal_per_sample_windows():
    dataset = WindowDataset(random_days(), window=20)
    indices = np.random.default_rng(1).choice(len(dataset), size=97, replace=False)  # unsorted, across days

    x, y = dataset.__getitems__(indices.tolist())
    expected = [dataset[i] for i in indices]
    torch.testing.assert_close(x, torch.stack([e[0] for e in expected]), rtol=0, atol=0)
    torch.testing.assert_close(y, torch.stack([e[1] for e in expected]), rtol=0, atol=0)


def test_loaders_batch_in_order_and_subsample_validation_deterministically():
    from deepofi.data_process.windows import make_loaders_from_days

    days = random_days(n_days=4)
    (train, val, test), _ = make_loaders_from_days(days[:2], days[2:3], days[3:], window=20, batch_size=64, max_val_samples=50)
    (_, val_again, _), _ = make_loaders_from_days(days[:2], days[2:3], days[3:], window=20, batch_size=64, max_val_samples=50)

    assert len(val.dataset) == 50 and list(val.dataset.indices) == list(val_again.dataset.indices)
    x_val = torch.cat([x for x, _ in val])
    assert x_val.shape == (50, 20, 5)
    x_test = torch.cat([x for x, _ in test])
    assert len(x_test) == len(test.dataset)  # the test set is never subsampled
    first_x, _ = test.dataset[0]
    torch.testing.assert_close(x_test[0], first_x)
    x_train, y_train = next(iter(train))
    assert x_train.shape == (64, 20, 5) and y_train.shape == (64, 4)
