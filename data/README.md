# Data

Data provided for free by IEX. By using it you agree to the [IEX Historical Data Terms of Use](https://www.iex.io/products/market-data-connectivity/hist-terms).

The full dataset (dataset v1, ~4 GB) is not committed. Only `samples/` is: the first 200 events of the 09:40–15:50 session for BAC on 20 July 2026, to show the format. Regenerate them with `python scripts/make_samples.py` (any `--ticker`, `--date`, `--rows`).

## Stored dataset

`book/{TICKER}/{YYYYMMDD}.parquet`, one row per order book state after a complete DEEP update (event flag 0x1), cleaned to the session (sample: `samples/BAC_20260720_book.csv`):

| Column | Meaning |
|---|---|
| `ts_event` | Event time, nanoseconds since the Unix epoch (UTC) |
| `bid_px_1..10`, `ask_px_1..10` | Price of the i-th best bid / ask level, USD; empty if the level does not exist |
| `bid_sz_1..10`, `ask_sz_1..10` | Displayed shares at that level; 0 if the level does not exist |
| `mid` | (ask_px_1 + bid_px_1) / 2 |
| `n_levels_bid`, `n_levels_ask` | Number of non-empty levels on each side (up to 10) |

`meta/{TICKER}/{YYYYMMDD}.json` holds the day's cleaning counts and characteristics: `updates` (book events kept), `price_changes` (mid-price changes, used for Δt), `trades`, `tick_size_frac` (share of time the spread is one tick), `median_spread_ticks`, and how many rows were dropped outside the session, during halts or as crossed/locked books.

## Model inputs and targets

Computed from the book on the fly by `deepofi.data_process.dataset`; the samples show them before winsorization and z-scoring (which are fitted on the training weeks). A model sees the last 100 rows as one window.

| File | Columns | Meaning |
|---|---|---|
| `samples/BAC_20260720_of.csv` | `bOF_1..10`, `aOF_1..10` | Order flow at each level (formulas (2)–(3) of the paper), shares |
| `samples/BAC_20260720_ofi.csv` | `OFI_1..10` | Order flow imbalance, bOF_i − aOF_i |
| `samples/BAC_20260720_lob.csv` | `ask_px_i`, `ask_sz_i`, `bid_px_i`, `bid_sz_i` for i = 1..10 | Raw book state in the paper's order; an empty level takes the price of the deepest non-empty level on that side and volume 0 |
| `samples/BAC_20260720_targets_short.csv` | `r_{h}dt`, `valid_{h}dt` | Mid-price return p(t + h·Δt) − p(t + 10 ms) in USD for h = 0.2, …, 2.0, and whether the horizon fits inside the session |

Δt for the sample is 831.5 ms: 2.34·10⁷ ms divided by the average number of mid-price changes per day over the training weeks of the window containing 20 July.
