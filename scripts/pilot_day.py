"""Pilot (M2): activity of every symbol on one IEX DEEP day, to choose the ticker universe.

Pass 1 counts book events and trades per symbol inside the trimmed session. Pass 2 rebuilds the
book for the most active symbols and adds mid-price changes, spread and tick-size statistics.

    python scripts/pilot_day.py data/raw/20261014_IEXTP1_DEEP1.0.pcap.gz --top 150
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
from deepofi.iex.deep import KIND_ASK, KIND_BID, KIND_TRADE, ParseStats, decode_symbol, encode_symbol, iter_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap", type=Path)
    parser.add_argument("--top", type=int, default=150, help="symbols to rebuild the book for")
    parser.add_argument("--batch", type=int, default=50, help="symbols per pass in the second stage")
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
    chunks = list(iter_records(path, symbols=symbols))
    records = np.concatenate(chunks)
    for symbol in symbols:
        own = records[records["symbol"] == np.uint64(encode_symbol(symbol))]
        book, report = clean_book(reconstruct_book(own), own, date)
        n_trades = int(((own["kind"] == KIND_TRADE) & (own["ts"] >= session_bounds(date)[0]) & (own["ts"] < end)).sum())
        row = {"symbol": symbol, **report}
        if len(book):
            row.update(day_characteristics(book, n_trades, end))
            row["mean_mid"] = float(np.nanmean(book["mid"]))
            row["spread_positive"] = bool((book["ask_px_1"] > book["bid_px_1"]).all())
        rows.append(row)
    return rows


def main():
    args = parse_args()
    date = re.match(r"(\d{8})", args.pcap.name).group(1)
    start, end = session_bounds(date)
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    events, trades, stats = count_activity(args.pcap, start, end)
    print(f"Pass 1: {time.time() - t0:.0f} s, {stats.packets:,} packets, gaps {stats.gap_messages}, duplicates {stats.duplicate_packets}")
    print("Messages by type:", dict(sorted(stats.counts_by_type().items(), key=lambda kv: -kv[1])))

    activity = pd.DataFrame({
        "symbol": [decode_symbol(s) for s in events],
        "book_events": list(events.values()),
    })
    activity["trades"] = [trades.get(s, 0) for s in events]
    activity = activity.sort_values("book_events", ascending=False).reset_index(drop=True)

    top = activity["symbol"].head(args.top).tolist()
    detail = []
    for i in range(0, len(top), args.batch):
        t1 = time.time()
        detail += book_statistics(args.pcap, top[i:i + args.batch], date, end)
        print(f"Pass 2, symbols {i + 1}-{min(i + args.batch, len(top))}: {time.time() - t1:.0f} s")

    activity = activity.merge(pd.DataFrame(detail).drop(columns="trades", errors="ignore"), on="symbol", how="left")
    path = args.out / f"activity_{date}.csv"
    activity.to_csv(path, index=False)
    print(f"Saved {path} ({len(activity)} symbols, total {time.time() - t0:.0f} s)")
    print(activity.head(30).to_string())


if __name__ == "__main__":
    main()
