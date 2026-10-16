import numpy as np
import torch

from deepofi.data_process.windows import WindowDataset
from deepofi.evaluation.metrics import calculate_metrics, daily_metrics


def test_daily_r2_uses_the_week_mean_as_benchmark():
    y_true = torch.tensor([[1.0], [3.0], [10.0], [14.0]])
    y_pred = torch.tensor([[2.0], [2.0], [12.0], [12.0]])
    days = np.array([0, 0, 1, 1])

    daily = daily_metrics(y_true, y_pred, days, ["20260824", "20260825"])

    week_mean = 7.0
    assert daily["20260824"]["n_obs"] == 2
    assert daily["20260824"]["mse"] == [1.0]
    np.testing.assert_allclose(daily["20260824"]["r2_os"], [1 - 1.0 / ((36 + 16) / 2)])
    np.testing.assert_allclose(daily["20260825"]["r2_os"], [1 - 4.0 / ((9 + 49) / 2)])
    # Against its own day mean the first day would score far worse: the benchmark choice matters.
    assert calculate_metrics(y_true[:2], y_pred[:2])["r2_os"][0] == 0.0
    assert week_mean == y_true.mean().item()


def test_window_index_gives_the_day_of_each_test_sample():
    days = [(np.zeros((5, 1), np.float32), np.zeros((5, 2), np.float32), np.ones((5, 2), bool)) for _ in range(2)]
    dataset = WindowDataset(days, window=3)
    assert dataset.index[:, 0].tolist() == [0, 0, 0, 1, 1, 1]
