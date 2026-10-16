"""Experiments beyond the paper, fixed in advance (no tuning on the test weeks).

    python scripts/extensions.py ensemble   # equal-weight average of ARX and CNN-LSTM forecasts, OF, 6 tickers
    python scripts/extensions.py pooled     # one ARX for all 11 stocks (universality), OFI

Results go to results/{name}/{input}/{ticker}/window_0N/metrics.csv in the usual format (weekly and
daily R2_OS), so they can be compared with the single models.
  ensemble: the trained weights of the member models are reused; predictions are averaged in dollars.
  pooled:   inputs and targets are normalized per stock as usual; the ridge sums of all stocks are
            added, the penalty is chosen on the pooled validation week, and each stock is tested alone.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from deepofi.config import load_config
from deepofi.data_process.dataset import make_window_loaders
from deepofi.data_process.splits import get_splits
from deepofi.data_process.targets import SHORT_HORIZONS
from deepofi.data_process.windows import INPUT_FEATURES
from deepofi.engine import load_weights
from deepofi.evaluation.metrics import calculate_metrics, daily_metrics, get_results_dir, predict, save_metrics
from deepofi.models import get_model
from deepofi.models.arx import TRAINING_PARAMS as ARX_PARAMS
from deepofi.models.arx import _Moments, fit_moments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NN_TICKERS = ["BAC", "VZ", "NKE", "USB", "OXY", "UBER"]
ALL_TICKERS = ["DIS", "CTSH", "UBER", "MO", "NEE", "OXY", "USB", "BSX", "NKE", "VZ", "BAC"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("ensemble", "pooled"))
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training jobs")
    return parser.parse_args()


def window_loaders(data_root, ticker, window, input_name, config):
    settings = config["training"]
    return make_window_loaders(
        data_root, ticker, window["val_dates"], window["train_dates"], window["test_dates"], input_name,
        config=config, window=settings["window"], batch_size=1024, seed=settings["seed"],
    )


def save(name, input_name, ticker, window, test_loader, y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    daily = daily_metrics(y_true, y_pred, test_loader.dataset.index[:, 0], window["test_dates"])
    save_metrics(name, metrics, SHORT_HORIZONS, input_name, ticker, window["name"], daily)
    return 100 * sum(metrics["r2_os"]) / len(metrics["r2_os"])


def run_ensemble(data_root, device, members=("arx", "cnn_lstm"), input_name="of"):
    config = load_config()
    name = "ens_" + "_".join(members)
    for ticker in NN_TICKERS:
        for window in get_splits():
            if (PROJECT_ROOT / "results" / name / input_name / ticker / window["name"] / "metrics.csv").exists():
                print(f"{ticker} {window['name']}: already done, skipping", flush=True)
                continue
            started = time.time()
            (_, _, test), y_normalizer, _ = window_loaders(data_root, ticker, window, input_name, config)
            forecasts = []
            for member in members:
                model, _ = get_model(member, n_features=INPUT_FEATURES[input_name])
                model = load_weights(model, PROJECT_ROOT / "weights" / member / input_name / ticker / f"{window['name']}.pt", device=device)
                y_true, y_pred = predict(model, test, device=device, y_normalizer=y_normalizer)
                forecasts.append(y_pred)
                week = calculate_metrics(y_true, y_pred)["r2_os"]
                print(f"{ticker} {window['name']} {member}: week R2_OS {100 * sum(week) / len(week):+.3f}%", flush=True)
            r2 = save(name, input_name, ticker, window, test, y_true, torch.stack(forecasts).mean(dim=0))
            print(f"{ticker} {window['name']} ensemble: week R2_OS {r2:+.3f}% ({time.time() - started:.0f} s)", flush=True)


def run_pooled(data_root, input_name="ofi"):
    config = load_config()
    name = "arx_pooled"
    for window in get_splits():
        started = time.time()
        train, val, tests = _Moments(), _Moments(), {}
        for ticker in ALL_TICKERS:
            (train_loader, val_loader, test_loader), y_normalizer, _ = window_loaders(data_root, ticker, window, input_name, config)
            train = train.merge(_Moments.from_loader(train_loader))
            val = val.merge(_Moments.from_loader(val_loader))
            tests[ticker] = (test_loader, y_normalizer)
            print(f"{window['name']} {ticker}: sums added ({time.time() - started:.0f} s)", flush=True)

        model, _ = get_model("arx", n_features=INPUT_FEATURES[input_name])
        model, history = fit_moments(model, train, val, ARX_PARAMS["ridge_grid"])
        for ticker, (test_loader, y_normalizer) in tests.items():
            y_true, y_pred = predict(model, test_loader, device="cpu", y_normalizer=y_normalizer)
            r2 = save(name, input_name, ticker, window, test_loader, y_true, y_pred)
            with (get_results_dir(name, input_name, ticker, window["name"]) / "history.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(f"{window['name']} {ticker} pooled ARX: week R2_OS {r2:+.3f}%", flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.experiment == "ensemble":
        run_ensemble(args.data_root, args.device)
    else:
        run_pooled(args.data_root)
