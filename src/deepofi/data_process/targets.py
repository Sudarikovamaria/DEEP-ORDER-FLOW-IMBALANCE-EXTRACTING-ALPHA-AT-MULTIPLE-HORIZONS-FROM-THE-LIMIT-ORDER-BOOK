"""Forward mid-price returns at multiple horizons (Section 3.2.1 of the paper, HANDOFF §6)."""
import numpy as np

SHORT_HORIZONS = tuple(k / 5 for k in range(1, 11))  # h_k = k * delta_t / 5
NS_PER_MS = 1_000_000


def count_price_changes(mid):
    """Number of nonzero mid-price changes, ignoring undefined mids (half-tick resolution is 0.005)."""
    mid = mid[~np.isnan(mid)]
    return int(np.count_nonzero(np.abs(np.diff(mid)) > 1e-7))


def delta_t_ms(price_changes_per_day, numerator_ms=2.34e7):
    """Average time between mid-price changes: numerator / mean daily number of changes.

    Must be computed on a past or training period only."""
    return numerator_ms / float(np.mean(price_changes_per_day))


def forward_returns(ts, mid, delta_t_ms, session_end, horizons=SHORT_HORIZONS, delta_tau_ms=10):
    """r[t, k] = p(t + h_k) - p(t + delta_tau) in dollars, p(s) = last mid with timestamp <= s.

    Returns float32 returns (NaN where invalid) and a bool mask, False when t + h_k reaches the
    session end or a mid is undefined.
    """
    ts = np.asarray(ts, dtype=np.int64)
    mid = np.asarray(mid, dtype=np.float64)
    offsets = np.round(np.asarray(horizons) * delta_t_ms * NS_PER_MS).astype(np.int64)

    start = mid[np.searchsorted(ts, ts + delta_tau_ms * NS_PER_MS, side="right") - 1]
    targets = ts[:, None] + offsets[None, :]
    end = mid[np.searchsorted(ts, targets, side="right") - 1]

    returns = end - start[:, None]
    mask = (targets < session_end) & ~np.isnan(returns)
    returns[~mask] = np.nan
    return returns.astype(np.float32), mask
