"""Quality report for the clean dataset: missing or halted days, overnight price jumps, activity.

    python scripts/check_dataset.py --tickers MO NEE DIS ...

Flags a ticker when a period day is missing or empty after cleaning, when the open differs from the
previous close by more than --max-gap (a split or another corporate action), or when a day's
delta t falls below the selection threshold.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deepofi.data_process.dataset import load_meta
from deepofi.data_process.splits import get_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--max-gap", type=float, default=0.15, help="max |open / previous close - 1|")
    parser.add_argument("--min-delta-t-ms", type=float, default=180.0)
    return parser.parse_args()


def ticker_days(root, ticker, dates):
    rows = []
    for date in dates:
        book_path = root / "book" / ticker / f"{date}.parquet"
        meta_path = root / "meta" / ticker / f"{date}.json"
        if not meta_path.exists():
            rows.append({"date": date, "missing": True})
            continue
        meta = load_meta(root, ticker, date)
        row = {"date": date, "missing": False, "events": meta.get("rows_kept", 0), "halts": meta.get("halts", 0),
               "price_changes": meta.get("price_changes"), "tick_size_frac": meta.get("tick_size_frac")}
        if book_path.exists() and row["events"]:
            mid = pd.read_parquet(book_path, columns=["mid"])["mid"].dropna().to_numpy()
            row.update({"open": mid[0], "close": mid[-1]})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    dates = sorted({d for w in get_splits() for part in ("val_dates", "train_dates", "test_dates") for d in w[part]})
    summary = []

    for ticker in args.tickers:
        days = ticker_days(args.root, ticker, dates)
        empty = days["missing"] | (days.get("events", 0) == 0)
        gaps = (days["open"] / days["close"].shift(1) - 1).abs()
        delta_t = 2.34e7 / days["price_changes"]
        flags = []
        if empty.any():
            flags.append(f"missing/empty: {', '.join(days.loc[empty, 'date'])}")
        if (days["halts"] > 0).any():
            flags.append(f"halts: {', '.join(days.loc[days['halts'] > 0, 'date'])}")
        if (gaps > args.max_gap).any():
            flags.append(f"price jumps: {', '.join(f'{d} ({g:+.0%})' for d, g in zip(days.loc[gaps > args.max_gap, 'date'], gaps[gaps > args.max_gap]))}")
        if (delta_t < args.min_delta_t_ms).any():
            flags.append(f"days with delta t < {args.min_delta_t_ms:.0f} ms: {int((delta_t < args.min_delta_t_ms).sum())}")
        summary.append({
            "ticker": ticker,
            "days": int((~empty).sum()),
            "events_per_day": int(days["events"].median()),
            "delta_t_ms_median": round(float(np.nanmedian(delta_t)), 0),
            "delta_t_ms_min": round(float(np.nanmin(delta_t)), 0),
            "tick_size_frac": round(float(days["tick_size_frac"].median()), 2),
            "max_overnight_gap": f"{np.nanmax(gaps):.1%}",
            "flags": "; ".join(flags) or "ok",
        })

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
