"""Build the clean dataset for selected tickers from IEX DEEP days (HANDOFF §3, §4).

    python scripts/build_dataset.py data/records/2026*.bin --tickers BAC NFLX
    python scripts/build_dataset.py data/raw/20261014_IEXTP1_DEEP1.0.pcap.gz --tickers BAC

Sources are extracted records (data/records/{date}.bin from overnight.py) or raw pcaps. For each
ticker and day it stores only the clean book (data/book/{TICKER}/{date}.parquet) and day
characteristics (data/meta/{TICKER}/{date}.json); inputs and targets are derived on the fly by
deepofi.data_process.dataset.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from deepofi.config import load_config, session_kwargs
from deepofi.data_process.book import reconstruct_book
from deepofi.data_process.characteristics import day_characteristics
from deepofi.data_process.clean import clean_book, session_bounds
from deepofi.iex.deep import KIND_TRADE, encode_symbol, load_records, read_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, nargs="+", help="data/records/{date}.bin files or raw pcap(.gz)")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline.yaml")
    return parser.parse_args()


def build_ticker(records, ticker, date, root, config):
    own = records[records["symbol"] == np.uint64(encode_symbol(ticker))]
    cleaning = config["cleaning"]
    session = session_kwargs(config)
    start, end = session_bounds(date, **session)

    book, report = clean_book(reconstruct_book(own, config["book"]["levels"]), own, date, cleaning["halt_policy"], cleaning["drop_unchanged"], **session)
    summary = {"date": date, **report}
    if len(book):
        trades = int(((own["kind"] == KIND_TRADE) & (own["ts"] >= start) & (own["ts"] < end)).sum())
        summary.update(day_characteristics(book, trades, end))
        (root / "book" / ticker).mkdir(parents=True, exist_ok=True)
        book.to_parquet(root / "book" / ticker / f"{date}.parquet", index=False)

    (root / "meta" / ticker).mkdir(parents=True, exist_ok=True)
    (root / "meta" / ticker / f"{date}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)

    for source in args.sources:
        date = re.match(r"(\d{8})", source.name).group(1)
        t0 = time.time()
        records = load_records(source, symbols=args.tickers) if source.suffix == ".bin" else read_records(source, symbols=args.tickers)
        for ticker in args.tickers:
            summary = build_ticker(records, ticker, date, args.root, config)
            if summary["rows_kept"]:
                print(f"{date} {ticker}: {summary['rows_kept']:,} events, {summary['price_changes']:,} mid changes "
                      f"(delta t {config['targets']['delta_t_numerator_ms'] / summary['price_changes']:.0f} ms), "
                      f"tick_size_frac {summary['tick_size_frac']:.2f}")
            else:
                print(f"{date} {ticker}: no rows left after cleaning {summary}")
        print(f"{date}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
