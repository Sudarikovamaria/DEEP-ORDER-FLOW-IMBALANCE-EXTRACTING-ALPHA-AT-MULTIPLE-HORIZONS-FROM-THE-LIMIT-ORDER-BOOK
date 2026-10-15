import numpy as np
import pandas as pd
import pytest

from deepofi.data_process.book import reconstruct_book
from deepofi.data_process.clean import clean_book, halt_intervals, session_bounds
from deepofi.iex.deep import KIND_ASK, KIND_BID, KIND_OPERATIONAL_HALT, KIND_TRADE, KIND_TRADING_STATUS, RECORD_DTYPE


def records(rows):
    """rows: (kind, flags, ts, price in dollars, size)"""
    out = np.zeros(len(rows), dtype=RECORD_DTYPE)
    for i, (kind, flags, ts, price, size) in enumerate(rows):
        out[i] = (ts, round(price * 10_000), 0, size, kind, flags)
    return out


def test_spec_example_atomic_update():
    # "Consuming Price Level Update Messages" example from the DEEP v1.08 specification.
    book = reconstruct_book(records([
        (KIND_ASK, 1, 1, 25.30, 100),
        (KIND_ASK, 1, 2, 25.20, 100),
        (KIND_ASK, 1, 3, 25.10, 100),
        (KIND_BID, 1, 4, 25.00, 100),
        (KIND_BID, 1, 5, 24.90, 100),
        (KIND_ASK, 0, 6, 25.10, 0),
        (KIND_ASK, 1, 6, 25.20, 0),
    ]))

    assert len(book) == 6
    assert book["ask_px_1"].iloc[4] == pytest.approx(25.10)
    last = book.iloc[-1]
    assert last["ask_px_1"] == pytest.approx(25.30) and np.isnan(last["ask_px_2"]) and last["ask_sz_2"] == 0
    assert (last["bid_px_1"], last["bid_px_2"]) == pytest.approx((25.00, 24.90))
    assert last["mid"] == pytest.approx(25.15)
    assert (last["n_levels_bid"], last["n_levels_ask"]) == (2, 1)


def test_deleting_best_level_promotes_eleventh():
    rows = [(KIND_BID, 1, i, 10.00 - 0.01 * i, 100 + i) for i in range(12)]
    rows.append((KIND_BID, 1, 12, 10.00, 0))
    book = reconstruct_book(records(rows))

    last = book.iloc[-1]
    assert last["bid_px_1"] == pytest.approx(9.99)
    assert last["bid_px_10"] == pytest.approx(9.90)
    assert last["bid_sz_10"] == 110
    assert last["n_levels_bid"] == 10
    assert np.isnan(last["mid"])


def test_size_update_and_trades_are_ignored_for_snapshots():
    book = reconstruct_book(records([
        (KIND_BID, 1, 1, 10.00, 100),
        (KIND_TRADE, 0, 2, 10.00, 40),
        (KIND_BID, 1, 2, 10.00, 60),
    ]))
    assert book["bid_sz_1"].tolist() == [100, 60]


def test_session_bounds_follow_dst():
    winter_start, winter_end = session_bounds("20260302")
    summer_start, _ = session_bounds("20260310")
    assert pd.Timestamp(winter_start, tz="UTC") == pd.Timestamp("2026-03-02 14:40", tz="UTC")
    assert pd.Timestamp(winter_end, tz="UTC") == pd.Timestamp("2026-03-02 20:50", tz="UTC")
    assert pd.Timestamp(summer_start, tz="UTC") == pd.Timestamp("2026-03-10 13:40", tz="UTC")


def session_book(date="20260302"):
    start, _ = session_bounds(date)
    minute = 60 * 10**9
    rows = [
        (KIND_BID, 1, start - minute, 10.00, 100),   # before 09:40, dropped
        (KIND_ASK, 1, start + minute, 10.02, 100),
        (KIND_BID, 1, start + 2 * minute, 10.01, 50),
        (KIND_BID, 1, start + 3 * minute, 10.03, 50),  # crossed
        (KIND_BID, 1, start + 4 * minute, 10.03, 0),   # back to the last kept state: unchanged
        (KIND_ASK, 1, start + 5 * minute, 10.05, 0),   # level that was never there: state unchanged
    ]
    return reconstruct_book(records(rows)), start, minute


def test_clean_book_steps():
    book, _, _ = session_book()
    cleaned, report = clean_book(book, records([]), "20260302")

    assert report == {
        "rows_total": 6, "outside_session": 1, "halts": 0, "halted": 0,
        "crossed_or_locked": 1, "unchanged": 2, "rows_kept": 2,
    }
    assert cleaned["bid_px_1"].iloc[-1] == pytest.approx(10.01)


def test_halt_policies():
    book, start, minute = session_book()
    status = records([
        (KIND_TRADING_STATUS, ord("T"), start - 60 * minute, 0, 0),
        (KIND_TRADING_STATUS, ord("H"), start + int(1.5 * minute), 0, 0),
        (KIND_TRADING_STATUS, ord("T"), start + int(2.5 * minute), 0, 0),
    ])

    assert halt_intervals(status) == [(start + int(1.5 * minute), start + int(2.5 * minute))]
    dropped_day, report = clean_book(book, status, "20260302")
    assert len(dropped_day) == 0 and report["halts"] == 1
    dropped_period, _ = clean_book(book, status, "20260302", halt_policy="drop_period")
    assert len(dropped_period) == 2


def test_operational_halt_until_end_of_day():
    status = records([(KIND_OPERATIONAL_HALT, ord("O"), 5, 0, 0)])
    assert halt_intervals(status, end_of_day=100) == [(5, 100)]
