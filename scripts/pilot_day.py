"""Pilot (M2): activity of every symbol on one IEX DEEP day, to choose the ticker universe.

Pass 1 counts book events and trades per symbol inside the trimmed session. Pass 2 rebuilds the
book for the selected symbols and adds mid-price changes, spread and tick-size statistics.
With --stocks-only, pass 2 covers every common stock with at least --min-events book events.

    python scripts/pilot_day.py data/raw/20260717_IEXTP1_DEEP1.0.pcap.gz --stocks-only
"""
import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from deepofi.data_process.book import reconstruct_book
from deepofi.data_process.characteristics import day_characteristics
from deepofi.data_process.clean import clean_book, session_bounds
from deepofi.data_process.universe import is_common_stock, load_symbol_directory
from deepofi.iex.deep import KIND_ASK, KIND_BID, KIND_TRADE, ParseStats, decode_symbol, encode_symbol, iter_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap", type=Path)
    parser.add_argument("--stocks-only", action="store_true", help="pass 2 for all common stocks above --min-events")
    parser.add_argument("--min-events", type=int, default=200_000)
    parser.add_argument("--top", type=int, default=150, help="without --stocks-only: most active symbols for pass 2")
    parser.add_argument("--batch", type=int, default=25, help="symbols per pass in the second stage")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "meta")
    return parser.parse_args()


def count_activity(path, start, end):
    events, trades = Counter(), Counter()
    stats = ParseStats()
    for chunk in iter_records(path, stats=stats):
        in_session = (chunk["ts"] >= start) & (chunk["ts"] < end)
        plu = in_session & ((chunk["kind"] == KIND_BID) | (chunk["kind"] == KIND_ASK)) & (chunk["flags"] == 1)
        for counter, selected in ((events, plu), (trades, in_session & (chunk["kind"] == KIND_TRADE))):
            symbols, counts = np.unique(chunk["symbol"][selected], return_counts=True)
            counter.update(dict(zip(symbols.tolist(), counts.tolist())))
    return events, trades, stats


def book_statistics(path, symbols, date, end):
    rows = []
    records = np.concatenate(list(iter_records(path, symbols=symbols)))
    for symbol in symbols:
        own = records[records["symbol"] == np.uint64(encode_symbol(symbol))]
        book, report = clean_book(reconstruct_book(own), own, date)
        n_trades = int(((own["kind"] == KIND_TRADE) & (own["ts"] >= session_bounds(date)[0]) & (own["ts"] < end)).sum())
        row = {"symbol": symbol, **report}
        if len(book):
            row.update(day_characteristics(book, n_trades, end))
            row["mean_mid"] = float(np.nanmean(book["mid"]))
        rows.append(row)
    return rows


def run_pilot(pcap, out_dir, stocks_only=True, min_events=200_000, top=150, batch=25, log=print):
    date = re.match(r"(\d{8})", Path(pcap).name).group(1)
    start, end = session_bounds(date)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    events, trades, stats = count_activity(pcap, start, end)
    log(f"Pass 1: {time.time() - t0:.0f} s, {stats.packets:,} packets, gaps {stats.gap_messages}, duplicates {stats.duplicate_packets}")
    log(f"Messages by type: {dict(sorted(stats.counts_by_type().items(), key=lambda kv: -kv[1]))}")

    activity = pd.DataFrame({"symbol": [decode_symbol(s) for s in events], "book_events": list(events.values())})
    activity["trades"] = [trades.get(s, 0) for s in events]
    activity["stock"] = is_common_stock(activity["symbol"], load_symbol_directory())
    activity = activity.sort_values("book_events", ascending=False).reset_index(drop=True)

    if stocks_only:
        selected = activity[activity["stock"] & (activity["book_events"] >= min_events)]["symbol"].tolist()
    else:
        selected = activity["symbol"].head(top).tolist()

    detail = []
    for i in range(0, len(selected), batch):
        t1 = time.time()
        detail += book_statistics(pcap, selected[i:i + batch], date, end)
        log(f"Pass 2, symbols {i + 1}-{min(i + batch, len(selected))} of {len(selected)}: {time.time() - t1:.0f} s")

    if detail:
        activity = activity.merge(pd.DataFrame(detail).drop(columns="trades", errors="ignore"), on="symbol", how="left")
    path = Path(out_dir) / f"activity_{date}.csv"
    activity.to_csv(path, index=False)
    log(f"Saved {path} ({len(activity)} symbols, {len(selected)} with book statistics, total {time.time() - t0:.0f} s)")
    return activity, selected


def main():
    args = parse_args()
    activity, _ = run_pilot(args.pcap, args.out, args.stocks_only, args.min_events, args.top, args.batch)
    print(activity.head(30).to_string())


if __name__ == "__main__":
    main()
