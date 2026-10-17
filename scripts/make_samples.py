"""Small CSV samples of every data format, to show what the dataset and the model inputs look like.

    python scripts/make_samples.py                     # BAC, 20260720, first 200 events of the session

Writes data/samples/: the stored clean book and the inputs (OF, OFI, LOB) and targets that the loader
derives from it on the fly. Data provided for free by IEX (see data/README.md).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deepofi.config import load_config
from deepofi.data_process.dataset import day_arrays, load_book, window_delta_t
from deepofi.data_process.splits import get_splits
from deepofi.data_process.targets import SHORT_HORIZONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEVELS = range(1, 11)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="BAC")
    parser.add_argument("--date", default="20260720")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "samples")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    book = load_book(args.data_root, args.ticker, args.date)
    window = next(w for w in get_splits() if args.date in w["val_dates"] + w["train_dates"] + w["test_dates"])
    dt = window_delta_t(args.data_root, args.ticker, window["train_dates"], config["targets"]["delta_t_numerator_ms"])
    ts = book["ts_event"].to_numpy()[: args.rows]
    prefix = args.out / f"{args.ticker}_{args.date}"

    book.head(args.rows).to_csv(f"{prefix}_book.csv", index=False)

    columns = {
        "of": [f"bOF_{i}" for i in LEVELS] + [f"aOF_{i}" for i in LEVELS],
        "ofi": [f"OFI_{i}" for i in LEVELS],
        "lob": [name for i in LEVELS for name in (f"ask_px_{i}", f"ask_sz_{i}", f"bid_px_{i}", f"bid_sz_{i}")],
    }
    for input_name, names in columns.items():
        x, y, mask = day_arrays(book, args.date, input_name, dt, "short", config)
        frame = pd.DataFrame(x[: args.rows], columns=names)
        frame.insert(0, "ts_event", ts)
        frame.to_csv(f"{prefix}_{input_name}.csv", index=False)

    horizons = [f"{h:.1f}" for h in SHORT_HORIZONS]
    targets = pd.DataFrame(np.round(y[: args.rows], 6), columns=[f"r_{h}dt" for h in horizons])
    targets[[f"valid_{h}dt" for h in horizons]] = mask[: args.rows]
    targets.insert(0, "ts_event", ts)
    targets.to_csv(f"{prefix}_targets_short.csv", index=False)

    print(f"Saved samples to {args.out} ({args.rows} events, delta t {dt:.1f} ms from the window's training days)")


if __name__ == "__main__":
    main()
