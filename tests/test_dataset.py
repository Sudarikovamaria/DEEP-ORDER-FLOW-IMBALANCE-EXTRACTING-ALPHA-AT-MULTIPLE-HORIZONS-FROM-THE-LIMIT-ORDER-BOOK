import json

import numpy as np

from deepofi.config import load_config
from deepofi.data_process.book import reconstruct_book
from deepofi.data_process.clean import session_bounds
from deepofi.data_process.dataset import day_arrays, load_book, make_window_loaders, window_delta_t
from deepofi.data_process.features import order_flow
from deepofi.data_process.targets import SHORT_HORIZONS, forward_returns
from deepofi.iex.deep import KIND_ASK, KIND_BID, RECORD_DTYPE


def random_book(date, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    start, _ = session_bounds(date)
    records = np.zeros(2 * n, dtype=RECORD_DTYPE)
    bid = 100_000
    for i in range(n):
        bid += int(rng.choice([-100, 0, 0, 0, 100]))
        ts = start + i * 50_000_000
        records[2 * i] = (ts, bid, 0, rng.integers(1, 9) * 100, KIND_BID, 1)
        records[2 * i + 1] = (ts, bid + 100, 0, rng.integers(1, 9) * 100, KIND_ASK, 1)
    return reconstruct_book(records)


def write_day(root, ticker, date, book, price_changes):
    (root / "book" / ticker).mkdir(parents=True, exist_ok=True)
    (root / "meta" / ticker).mkdir(parents=True, exist_ok=True)
    book.to_parquet(root / "book" / ticker / f"{date}.parquet", index=False)
    (root / "meta" / ticker / f"{date}.json").write_text(json.dumps({"date": date, "price_changes": price_changes}))


def test_day_arrays_match_direct_computation(tmp_path):
    config = load_config()
    date = "20260721"
    write_day(tmp_path, "XYZ", date, random_book(date), 1000)
    book = load_book(tmp_path, "XYZ", date)

    x, y, mask = day_arrays(book, date, "of", 200.0, "short", config)

    np.testing.assert_array_equal(x, order_flow(book))
    expected_y, expected_mask = forward_returns(book["ts_event"].to_numpy(), book["mid"].to_numpy(), 200.0, session_bounds(date)[1], SHORT_HORIZONS, 10)
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(mask, expected_mask)
    assert day_arrays(book, date, "lob", 200.0, "long", config)[0].shape[1] == 40
    assert day_arrays(book, date, "ofi", 200.0, "long", config)[1].shape[1] == len(config["targets"]["long_horizons"])


def test_window_delta_t_uses_training_days_only(tmp_path):
    dates = ["20260720", "20260721", "20260722", "20260723"]
    for i, date in enumerate(dates):
        write_day(tmp_path, "XYZ", date, random_book(date, n=400, seed=i), [10, 1000, 3000, 999_999][i])

    assert window_delta_t(tmp_path, "XYZ", dates[1:3], 2.34e7) == 2.34e7 / 2000
    (train, val, test), y_normalizer, dt = make_window_loaders(tmp_path, "XYZ", dates[:1], dates[1:3], dates[3:], window=50)
    assert dt == 2.34e7 / 2000
    x, y = next(iter(train))
    assert x.shape[1:] == (50, 20) and y.shape[1] == 10
