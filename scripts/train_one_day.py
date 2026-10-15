"""Smoke test on one real day: intraday train / val / test split (not the paper's protocol).

    python scripts/train_one_day.py --model arx --ticker BAC --date 20261014

The day is cut at --splits (New York time) into train, val and test. Targets whose horizon
reaches into the next segment are masked, so no segment sees prices from a later one.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deepofi.data_process.targets import NS_PER_MS, SHORT_HORIZONS
from deepofi.data_process.windows import INPUT_FEATURES, make_loaders_from_days
from deepofi.engine import train
from deepofi.evaluation.metrics import calculate_metrics, predict
from deepofi.models import MODELS, get_fit, get_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--input", choices=INPUT_FEATURES, default="of")
    parser.add_argument("--splits", default="13:30,14:30", help="train/val and val/test cut times, New York")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=200_000, help="windows per epoch for SGD models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_segments(root, ticker, date, input_name, splits):
    features = np.load(root / "features" / ticker / f"{date}_{input_name}.npy")
    ts = np.load(root / "features" / ticker / f"{date}_ts.npy")
    targets = np.load(root / "targets" / ticker / f"{date}_short.npy")
    mask = np.load(root / "targets" / ticker / f"{date}_mask_short.npy")
    meta = json.loads((root / "meta" / ticker / f"{date}.json").read_text())

    offsets = np.round(np.array(SHORT_HORIZONS) * meta["delta_t_ms"] * NS_PER_MS).astype(np.int64)
    cuts = [pd.Timestamp(f"{date} {time}", tz="America/New_York").value for time in splits.split(",")]
    bounds = [ts[0]] + cuts + [ts[-1] + 1]

    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        rows = (ts >= start) & (ts < end)
        inside = ts[rows, None] + offsets[None, :] < end
        segments.append((features[rows], targets[rows], mask[rows] & inside))
    return segments, meta


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    (train_day, val_day, test_day), meta = load_segments(args.root, args.ticker, args.date, args.input, args.splits)
    fit = get_fit(args.model)

    (train_loader, val_loader, test_loader), y_normalizer = make_loaders_from_days(
        [train_day], [val_day], [test_day],
        batch_size=1024 if fit else 256,
        max_samples_per_epoch=None if fit else args.max_samples,
        seed=args.seed,
    )
    print(f"{args.ticker} {args.date}: delta_t {meta['delta_t_ms']:.0f} ms"
          f"{' (same day, smoke test only)' if meta.get('delta_t_same_day') else ''}; "
          f"windows train {len(train_loader.dataset):,} / val {len(val_loader.dataset):,} / test {len(test_loader.dataset):,}")

    model, params = get_model(args.model, n_features=INPUT_FEATURES[args.input])
    if fit:
        model, _ = fit(model, train_loader, val_loader, **params)
    else:
        model, _ = train(
            model, train_loader, val_loader,
            learning_rate=args.learning_rate or params["learning_rate"],
            weights_path=f"{args.model}_{args.input}_{args.ticker}_{args.date}_smoke.pt",
            epochs=args.epochs or params.get("epochs", 50),
            patience=params.get("patience", 5),
            device=args.device,
            save_weights=False,
        )

    y_true, y_pred = predict(model, test_loader, device="cpu" if fit else args.device)
    y_pred = torch.from_numpy(y_normalizer.inverse(y_pred.double().numpy()))
    metrics = calculate_metrics(y_true, y_pred)
    print("horizon  h, ms   R2_OS, %")
    for k, (multiple, r2) in enumerate(zip(SHORT_HORIZONS, metrics["r2_os"]), start=1):
        print(f"{k:7d}  {multiple * meta['delta_t_ms']:5.0f}  {100 * r2:8.3f}")
    print(f"mean R2_OS, %: {100 * np.mean(metrics['r2_os']):.3f}")


if __name__ == "__main__":
    main()
