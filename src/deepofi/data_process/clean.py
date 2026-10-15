"""Book cleaning, the analogue of Appendix A of the paper (HANDOFF §4)."""
import numpy as np
import pandas as pd

from ..iex.deep import KIND_OPERATIONAL_HALT, KIND_TRADING_STATUS

LEVELS = 10


def session_bounds(date, timezone="America/New_York", open_time="09:30", close_time="16:00", trim_minutes=10):
    """UTC nanosecond bounds [start, end) of the trimmed regular session, DST-aware."""
    trim = pd.Timedelta(minutes=trim_minutes)
    start = pd.Timestamp(f"{date} {open_time}", tz=timezone) + trim
    end = pd.Timestamp(f"{date} {close_time}", tz=timezone) - trim
    return start.value, end.value


def halt_intervals(records, end_of_day=np.iinfo(np.int64).max):
    """[start, end) intervals when the symbol was not trading: Trading Status other than 'T'
    or an IEX operational halt."""
    statuses = records[(records["kind"] == KIND_TRADING_STATUS) | (records["kind"] == KIND_OPERATIONAL_HALT)]
    halted = {KIND_TRADING_STATUS: False, KIND_OPERATIONAL_HALT: False}
    intervals = []
    start = None

    for kind, flags, ts in zip(statuses["kind"], statuses["flags"], statuses["ts"]):
        halted[kind] = flags != (ord("T") if kind == KIND_TRADING_STATUS else ord("N"))
        if any(halted.values()) and start is None:
            start = ts
        elif not any(halted.values()) and start is not None:
            intervals.append((start, ts))
            start = None

    if start is not None:
        intervals.append((start, end_of_day))
    return intervals


def clean_book(book, records, date, halt_policy="drop_day", drop_unchanged=True, **session):
    """Returns the cleaned book and a dict with the number of rows removed at each step.

    halt_policy: "drop_day" returns an empty book if a halt overlaps the session,
    "drop_period" removes only the rows inside halts.
    """
    start, end = session_bounds(date, **session)
    report = {"rows_total": len(book)}

    book = book[(book["ts_event"] >= start) & (book["ts_event"] < end)]
    report["outside_session"] = report["rows_total"] - len(book)

    halts = [(a, b) for a, b in halt_intervals(records) if a < end and b > start]
    report["halts"] = len(halts)
    n = len(book)
    if halts and halt_policy == "drop_day":
        book = book.iloc[:0]
    elif halts and halt_policy == "drop_period":
        inside = np.zeros(len(book), dtype=bool)
        for a, b in halts:
            inside |= (book["ts_event"].to_numpy() >= a) & (book["ts_event"].to_numpy() < b)
        book = book[~inside]
    elif halts:
        raise ValueError(f"Unknown halt_policy {halt_policy!r}")
    report["halted"] = n - len(book)

    n = len(book)
    book = book[~(book["bid_px_1"] >= book["ask_px_1"])]
    report["crossed_or_locked"] = n - len(book)

    n = len(book)
    if drop_unchanged and len(book):
        state = book[[f"{side}_{field}_{i}" for side in ("bid", "ask") for field in ("px", "sz") for i in range(1, LEVELS + 1)]]
        state = np.nan_to_num(state.to_numpy(dtype=np.float64), nan=0.0)
        changed = np.ones(len(state), dtype=bool)
        changed[1:] = (state[1:] != state[:-1]).any(axis=1)
        book = book[changed]
    report["unchanged"] = n - len(book)

    report["rows_kept"] = len(book)
    return book.reset_index(drop=True), report
