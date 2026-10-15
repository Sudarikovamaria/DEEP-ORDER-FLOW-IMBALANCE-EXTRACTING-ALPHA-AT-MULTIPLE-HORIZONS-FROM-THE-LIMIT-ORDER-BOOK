"""Build book / features / targets for selected tickers from one IEX DEEP day (HANDOFF §3).

    python scripts/build_dataset.py data/raw/20261014_IEXTP1_DEEP1.0.pcap.gz --tickers BAC AMZN NFLX

Delta t must come from a reference period before the sample (data/meta/{TICKER}.json,
key "delta_t_ms"). `--delta-t-same-day` computes it from the day itself, which looks ahead and
is only acceptable for smoke tests.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import yaml

from deepofi.data_process.book import reconstruct_book
from deepofi.data_process.characteristics import day_characteristics
from deepofi.data_process.clean import clean_book, session_bounds
from deepofi.data_process.features import limit_order_book, order_flow, order_flow_imbalance
from deepofi.data_process.targets import SHORT_HORIZONS, count_price_changes, delta_t_ms, forward_returns
from deepofi.iex.deep import KIND_TRADE, encode_symbol, read_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap", type=Path)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--delta-t-same-day", action="store_true", help="smoke tests only: delta t from this day")
    return parser.parse_args()


def reference_delta_t(root, ticker):
    path = root / "meta" / f"{ticker}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("delta_t_ms")


def build_ticker(records, ticker, date, root, config, delta_t_same_day):
    own = records[records["symbol"] == np.uint64(encode_symbol(ticker))]
    cleaning = config["cleaning"]
    session = config["session"]
    session_kwargs = {"timezone": session["timezone"], "open_time": session["open"], "close_time": session["close"], "trim_minutes": session["trim_minutes"]}
    start, end = session_bounds(date, **session_kwargs)

    book, report = clean_book(reconstruct_book(own, config["book"]["levels"]), own, date, cleaning["halt_policy"], cleaning["drop_unchanged"], **session_kwargs)
    if not len(book):
        print(f"{ticker}: no rows left after cleaning {report}")
        return None

    trades = int(((own["kind"] == KIND_TRADE) & (own["ts"] >= start) & (own["ts"] < end)).sum())
    day = day_characteristics(book, trades, end)

    targets_config = config["targets"]
    if delta_t_same_day:
        dt = delta_t_ms([count_price_changes(book["mid"].to_numpy())], targets_config["delta_t_numerator_ms"])
    else:
        dt = reference_delta_t(root, ticker)
        if dt is None:
            raise SystemExit(f"{ticker}: no delta_t_ms in data/meta/{ticker}.json; build the reference period first")

    for sub in ("book", "features", "targets"):
        (root / sub / ticker).mkdir(parents=True, exist_ok=True)
    book.to_parquet(root / "book" / ticker / f"{date}.parquet", index=False)

    of = order_flow(book, variant=config["features"]["of_variant"])
    features = {"of": of, "lob": limit_order_book(book), "ofi": order_flow_imbalance(of), "ts": book["ts_event"].to_numpy()}
    for name, array in features.items():
        np.save(root / "features" / ticker / f"{date}_{name}.npy", array)

    for name, horizons in (("short", SHORT_HORIZONS), ("long", targets_config["long_horizons"])):
        returns, mask = forward_returns(book["ts_event"].to_numpy(), book["mid"].to_numpy(), dt, end, horizons, targets_config["delta_tau_ms"])
        np.save(root / "targets" / ticker / f"{date}_{name}.npy", returns)
        np.save(root / "targets" / ticker / f"{date}_mask_{name}.npy", mask)

    summary = {"date": date, "delta_t_ms": dt, "delta_t_same_day": delta_t_same_day, **report, **day}
    (root / "meta" / ticker).mkdir(parents=True, exist_ok=True)
    (root / "meta" / ticker / f"{date}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    date = re.match(r"(\d{8})", args.pcap.name).group(1)

    t0 = time.time()
    records = read_records(args.pcap, symbols=args.tickers)
    print(f"Parsed {len(records):,} records for {len(args.tickers)} tickers in {time.time() - t0:.0f} s")

    for ticker in args.tickers:
        summary = build_ticker(records, ticker, date, args.root, config, args.delta_t_same_day)
        if summary:
            print(f"{ticker}: {summary['rows_kept']:,} events, delta_t {summary['delta_t_ms']:.0f} ms, "
                  f"tick_size_frac {summary['tick_size_frac']:.2f}, {summary['trades']:,} trades")


if __name__ == "__main__":
    main()
