"""Model inputs and targets computed on the fly from the stored clean books (data/book).

The frozen dataset is only data/book/{TICKER}/{date}.parquet plus data/meta/{TICKER}/{date}.json;
OF, LOB, OFI and forward returns are derived from it when a window is loaded, with delta t
estimated on the window's training days only.
"""
import json
from pathlib import Path

import pandas as pd

from ..config import load_config, session_kwargs
from .clean import session_bounds
from .features import limit_order_book, order_flow, order_flow_imbalance
from .targets import SHORT_HORIZONS, delta_t_ms, forward_returns
from .windows import make_loaders_from_days


def load_book(root, ticker, date):
    return pd.read_parquet(Path(root) / "book" / ticker / f"{date}.parquet")


def load_meta(root, ticker, date):
    return json.loads((Path(root) / "meta" / ticker / f"{date}.json").read_text())


def window_delta_t(root, ticker, dates, numerator_ms):
    """Delta t from the mean number of mid-price changes on `dates` (the training days)."""
    return delta_t_ms([load_meta(root, ticker, date)["price_changes"] for date in dates], numerator_ms)


def input_features(book, input_name, of_variant="paper"):
    if input_name == "lob":
        return limit_order_book(book)
    of = order_flow(book, variant=of_variant)
    if input_name == "of":
        return of
    if input_name == "ofi":
        return order_flow_imbalance(of)
    raise ValueError(f"Unknown input {input_name!r}")


def horizon_multiples(targets, config):
    if targets == "short":
        return SHORT_HORIZONS
    if targets == "long":
        return tuple(config["targets"]["long_horizons"])
    raise ValueError(f"Unknown target set {targets!r}")


def day_arrays(book, date, input_name, dt_ms, targets="short", config=None):
    """(x, y, mask) for one day: inputs [T, F], forward returns [T, H] in dollars, validity mask."""
    config = config or load_config()
    _, end = session_bounds(date, **session_kwargs(config))
    x = input_features(book, input_name, config["features"]["of_variant"])
    y, mask = forward_returns(
        book["ts_event"].to_numpy(), book["mid"].to_numpy(), dt_ms, end,
        horizon_multiples(targets, config), config["targets"]["delta_tau_ms"],
    )
    return x, y, mask


def make_window_loaders(root, ticker, val_dates, train_dates, test_dates, input_name="of", targets="short", config=None, **loader_kwargs):
    """Loaders for one rolling window. Returns (train, val, test), y normalizer and delta t (ms)."""
    config = config or load_config()
    dt = window_delta_t(root, ticker, train_dates, config["targets"]["delta_t_numerator_ms"])

    def days(dates):
        return [day_arrays(load_book(root, ticker, date), date, input_name, dt, targets, config) for date in dates]

    loaders, y_normalizer = make_loaders_from_days(days(train_dates), days(val_dates), days(test_dates), **loader_kwargs)
    return loaders, y_normalizer, dt
