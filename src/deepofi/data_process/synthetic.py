"""Synthetic dataset in the same file layout as the real one (HANDOFF §3).

Lets model owners develop and debug before the IEX dataset is ready. Targets are a linear
function of recent level-1 OFI plus noise, so a working model must reach R²_OS > 0, with a
hump-shaped term structure across horizons similar to the paper.

    python -m deepofi.data_process.synthetic
"""
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_ROOT = PROJECT_ROOT / "data" / "synthetic"
LEVELS = 10
TICK = 0.01


def business_days(start, n):
    days = np.busday_offset(np.datetime64(start, "D"), np.arange(n), roll="forward")
    return [str(day).replace("-", "") for day in days]


def make_day(rng, n_events, n_horizons=10, signal_lags=50, noise=1.5):
    bid_of = np.round(rng.standard_t(3, size=(n_events, LEVELS)) * 100)
    ask_of = np.round(rng.standard_t(3, size=(n_events, LEVELS)) * 100)
    of = np.concatenate([bid_of, ask_of], axis=1)
    ofi = bid_of - ask_of

    weights = np.exp(-np.arange(signal_lags) / 15.0)
    signal = np.convolve(ofi[:, 0], weights)[:n_events]
    signal /= signal.std()
    k = np.arange(1, n_horizons + 1)
    loadings = k * np.exp(-k / 4.0) * 0.01
    targets = signal[:, None] * loadings + rng.normal(scale=noise * 0.01, size=(n_events, n_horizons))

    mid = 100 + np.cumsum(rng.choice([-TICK, 0.0, TICK], size=n_events, p=[0.05, 0.9, 0.05]))
    lob = np.empty((n_events, 4 * LEVELS))
    for i in range(LEVELS):
        lob[:, 4 * i] = mid + (i + 0.5) * TICK
        lob[:, 4 * i + 1] = rng.lognormal(5, 1, n_events).round()
        lob[:, 4 * i + 2] = mid - (i + 0.5) * TICK
        lob[:, 4 * i + 3] = rng.lognormal(5, 1, n_events).round()

    # Horizons that run past the end of the session are invalid, as in the real targets.
    mask = np.arange(n_events)[:, None] < n_events - 20 * k

    ts = np.datetime64("2026-01-01T14:40", "ns").astype(np.int64) + np.cumsum(rng.exponential(1e8, n_events)).astype(np.int64)
    return {
        "of": of.astype(np.float32),
        "lob": lob.astype(np.float32),
        "ofi": ofi.astype(np.float32),
        "ts": ts,
        "short": targets.astype(np.float32),
        "mask_short": mask,
    }


def generate(root=SYNTHETIC_ROOT, tickers=("SYN1", "SYN2"), n_days=8, n_events=20_000, start="2026-01-05", seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    dates = business_days(start, n_days)

    for ticker in tickers:
        (root / "features" / ticker).mkdir(parents=True, exist_ok=True)
        (root / "targets" / ticker).mkdir(parents=True, exist_ok=True)
        for date in dates:
            day = make_day(rng, n_events)
            for name in ("of", "lob", "ofi", "ts"):
                np.save(root / "features" / ticker / f"{date}_{name}.npy", day[name])
            for name in ("short", "mask_short"):
                np.save(root / "targets" / ticker / f"{date}_{name}.npy", day[name])

    print(f"Synthetic data: {root} ({len(tickers)} tickers x {n_days} days x {n_events} events)")
    return dates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=SYNTHETIC_ROOT)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--events", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(root=args.root, n_days=args.days, n_events=args.events, seed=args.seed)
