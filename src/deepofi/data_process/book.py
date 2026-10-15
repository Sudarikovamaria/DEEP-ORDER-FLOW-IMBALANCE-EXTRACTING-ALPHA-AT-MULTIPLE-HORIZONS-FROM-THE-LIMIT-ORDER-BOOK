"""Top-of-book reconstruction from DEEP price level updates (HANDOFF §3, §4).

A snapshot is taken only after a price level update with event flag 1, so the in-transition
states of an atomic multi-level update are never seen (DEEP v1.08, "Consuming Price Level
Update Messages").
"""
import numba
import numpy as np
import pandas as pd

from ..iex.deep import KIND_ASK, KIND_BID, PRICE_SCALE

LEVELS = 10
_MAX_PRICE_LEVELS = 1 << 14


@numba.njit(cache=True)
def _apply(prices, sizes, n, price, size, descending):
    """Set `price` to `size` in a side kept sorted best-first; size 0 removes the level."""
    i = 0
    while i < n and (prices[i] > price if descending else prices[i] < price):
        i += 1
    found = i < n and prices[i] == price
    if size == 0:
        if found:
            prices[i:n - 1] = prices[i + 1:n].copy()
            sizes[i:n - 1] = sizes[i + 1:n].copy()
            n -= 1
        return n
    if found:
        sizes[i] = size
        return n
    if n == prices.shape[0]:
        raise ValueError("Too many price levels on one side")
    prices[i + 1:n + 1] = prices[i:n].copy()
    sizes[i + 1:n + 1] = sizes[i:n].copy()
    prices[i] = price
    sizes[i] = size
    return n + 1


@numba.njit(cache=True)
def _reconstruct(kind, flags, ts, price, size, levels):
    bid_prices = np.zeros(_MAX_PRICE_LEVELS, dtype=np.int64)
    bid_sizes = np.zeros(_MAX_PRICE_LEVELS, dtype=np.int64)
    ask_prices = np.zeros(_MAX_PRICE_LEVELS, dtype=np.int64)
    ask_sizes = np.zeros(_MAX_PRICE_LEVELS, dtype=np.int64)
    n_bid = 0
    n_ask = 0

    n_snapshots = 0
    for i in range(kind.shape[0]):
        if (kind[i] == 0 or kind[i] == 1) and flags[i] == 1:
            n_snapshots += 1

    out_ts = np.empty(n_snapshots, dtype=np.int64)
    out_bid_px = np.zeros((n_snapshots, levels), dtype=np.int64)
    out_bid_sz = np.zeros((n_snapshots, levels), dtype=np.int64)
    out_ask_px = np.zeros((n_snapshots, levels), dtype=np.int64)
    out_ask_sz = np.zeros((n_snapshots, levels), dtype=np.int64)
    out_n_bid = np.zeros(n_snapshots, dtype=np.int64)
    out_n_ask = np.zeros(n_snapshots, dtype=np.int64)

    j = 0
    for i in range(kind.shape[0]):
        if kind[i] == 0:
            n_bid = _apply(bid_prices, bid_sizes, n_bid, price[i], size[i], True)
        elif kind[i] == 1:
            n_ask = _apply(ask_prices, ask_sizes, n_ask, price[i], size[i], False)
        else:
            continue
        if flags[i] != 1:
            continue

        out_ts[j] = ts[i]
        k_bid = min(n_bid, levels)
        k_ask = min(n_ask, levels)
        out_bid_px[j, :k_bid] = bid_prices[:k_bid]
        out_bid_sz[j, :k_bid] = bid_sizes[:k_bid]
        out_ask_px[j, :k_ask] = ask_prices[:k_ask]
        out_ask_sz[j, :k_ask] = ask_sizes[:k_ask]
        out_n_bid[j] = k_bid
        out_n_ask[j] = k_ask
        j += 1

    return out_ts, out_bid_px, out_bid_sz, out_ask_px, out_ask_sz, out_n_bid, out_n_ask


def reconstruct_book(records, levels=LEVELS):
    """Book snapshots for one symbol from its records in feed order (iex.deep.RECORD_DTYPE).

    Columns follow the book parquet contract: empty levels have NaN price and size 0, `mid` is
    NaN when a side is empty, n_levels_* count non-empty levels among the top `levels`.
    """
    is_level_update = (records["kind"] == KIND_BID) | (records["kind"] == KIND_ASK)
    updates = records[is_level_update]
    ts, bid_px, bid_sz, ask_px, ask_sz, n_bid, n_ask = _reconstruct(
        updates["kind"], updates["flags"], updates["ts"], updates["price"], updates["size"].astype(np.int64), levels
    )

    columns = {"ts_event": ts}
    for side, px, sz in (("bid", bid_px, bid_sz), ("ask", ask_px, ask_sz)):
        prices = np.where(sz > 0, px / PRICE_SCALE, np.nan)
        for i in range(levels):
            columns[f"{side}_px_{i + 1}"] = prices[:, i]
        for i in range(levels):
            columns[f"{side}_sz_{i + 1}"] = sz[:, i].astype(np.int32)

    book = pd.DataFrame(columns)
    order = ["ts_event"] + [f"bid_px_{i}" for i in range(1, levels + 1)] + [f"bid_sz_{i}" for i in range(1, levels + 1)]
    order += [f"ask_px_{i}" for i in range(1, levels + 1)] + [f"ask_sz_{i}" for i in range(1, levels + 1)]
    book = book[order]
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2
    book["n_levels_bid"] = n_bid.astype(np.int8)
    book["n_levels_ask"] = n_ask.astype(np.int8)
    return book
