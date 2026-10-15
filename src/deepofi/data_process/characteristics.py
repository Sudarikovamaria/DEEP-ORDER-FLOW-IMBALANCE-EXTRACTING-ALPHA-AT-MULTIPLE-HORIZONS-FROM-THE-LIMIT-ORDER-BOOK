"""Per-day stock characteristics for meta json and the cross-sectional regressions (Table 5)."""
import numpy as np

from .targets import count_price_changes, delta_t_ms

TICK = 0.01


def day_characteristics(book, n_trades, session_end, tick=TICK):
    """book: cleaned book of one day; n_trades: trades inside the same session window."""
    ts = book["ts_event"].to_numpy()
    spread = (book["ask_px_1"] - book["bid_px_1"]).to_numpy()
    durations = np.diff(np.append(ts, session_end)).astype(np.float64)
    defined = ~np.isnan(spread)
    one_tick = defined & (spread < 1.5 * tick)

    return {
        "updates": int(len(book)),
        "price_changes": count_price_changes(book["mid"].to_numpy()),
        "trades": int(n_trades),
        "tick_size_frac": float(durations[one_tick].sum() / durations[defined].sum()) if defined.any() else float("nan"),
        "median_spread_ticks": float(np.median(spread[defined]) / tick) if defined.any() else float("nan"),
    }


def summarize(days, numerator_ms=2.34e7):
    """meta json fields from a list of day_characteristics() over the reference period."""
    updates = np.mean([d["updates"] for d in days])
    trades = np.mean([d["trades"] for d in days])
    price_changes = np.mean([d["price_changes"] for d in days])
    return {
        "delta_t_ms": delta_t_ms([d["price_changes"] for d in days], numerator_ms),
        "N_pricechg_per_day": float(price_changes),
        "updates_per_day": float(updates),
        "trades_per_day": float(trades),
        "tick_size_frac": float(np.nanmean([d["tick_size_frac"] for d in days])),
        "log_updates": float(np.log(updates)),
        "log_trades": float(np.log(trades)),
        "log_pricechg": float(np.log(price_changes)),
        "log_updates_over_pricechg": float(np.log(updates / price_changes)),
        "n_days": len(days),
    }
