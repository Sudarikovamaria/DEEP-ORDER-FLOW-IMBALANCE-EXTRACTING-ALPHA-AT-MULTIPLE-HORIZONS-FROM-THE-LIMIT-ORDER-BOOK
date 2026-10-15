import numpy as np

from deepofi.data_process.targets import count_price_changes, delta_t_ms, forward_returns

MS = 1_000_000


def test_forward_returns_use_last_mid_at_or_before_horizon():
    ts = np.array([0, 5, 20, 100, 250]) * MS
    mid = np.array([10.00, 10.01, 10.02, 10.05, 10.04])
    # delta_t = 100 ms, horizons 0.5 and 1.5 -> 50 ms and 150 ms; delta_tau = 10 ms.
    returns, mask = forward_returns(ts, mid, delta_t_ms=100, session_end=10_000 * MS, horizons=(0.5, 1.5))

    # t=0: p(10ms) = 10.01 (row at 5 ms), p(50ms) = 10.02, p(150ms) = 10.05
    np.testing.assert_allclose(returns[0], [0.01, 0.04], atol=1e-6)
    # t=100ms: p(110ms) = 10.05, p(150ms) = 10.05, p(250ms) = 10.04 (exactly at the horizon)
    np.testing.assert_allclose(returns[3], [0.0, -0.01], atol=1e-6)
    assert mask.all()


def test_mask_at_session_end_and_undefined_mid():
    ts = np.array([0, 100, 200]) * MS
    mid = np.array([10.0, np.nan, 10.1])
    returns, mask = forward_returns(ts, mid, delta_t_ms=100, session_end=250 * MS, horizons=(0.5, 1.0))

    assert mask.tolist() == [[True, False], [False, False], [False, False]]
    assert np.isnan(returns[~mask]).all()


def test_delta_t():
    assert count_price_changes(np.array([1.0, 1.0, 1.01, np.nan, 1.01, 1.0])) == 2
    assert delta_t_ms([1000, 3000]) == 2.34e7 / 2000


def test_day_characteristics():
    import pandas as pd

    from deepofi.data_process.characteristics import day_characteristics, summarize

    book = pd.DataFrame({
        "ts_event": [0, 10, 40],
        "bid_px_1": [10.00, 10.00, 10.01],
        "ask_px_1": [10.01, 10.03, 10.02],
    })
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2
    day = day_characteristics(book, n_trades=7, session_end=100)

    assert day["updates"] == 3 and day["price_changes"] == 1 and day["trades"] == 7  # 10.005 -> 10.015 -> 10.015
    assert abs(day["tick_size_frac"] - (10 + 60) / 100) < 1e-12
    meta = summarize([day, day])
    assert meta["delta_t_ms"] == 2.34e7 and meta["n_days"] == 2
