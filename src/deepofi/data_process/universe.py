"""Ticker universe: common stocks only, selected by activity and delta t (DEVIATIONS.md, row 8).

The symbol directory is Nasdaq Trader's nasdaqtraded.txt (all US-listed securities with an ETF flag).
"""
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYMBOL_DIRECTORY = PROJECT_ROOT / "data" / "meta" / "nasdaqtraded.txt"
SYMBOL_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
NOT_COMMON = r"\b(?:Fund|Trust|ETN|ETF|Notes?|Warrants?|Preferred|Units?|Rights?|Depositary Receipt)\b"


def load_symbol_directory(path=SYMBOL_DIRECTORY, url=SYMBOL_DIRECTORY_URL):
    """The symbol directory; downloaded to `path` first if it is not there yet."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    directory = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)
    directory = directory[~directory["Nasdaq Traded"].str.startswith("File Creation Time")]
    directory = directory[directory["Test Issue"] == "N"].drop_duplicates("Symbol")
    return directory.set_index("Symbol")


def is_common_stock(symbols, directory):
    symbols = pd.Series(list(symbols))
    names = symbols.map(directory["Security Name"])
    common = (symbols.map(directory["ETF"]) == "N") & ~names.str.contains(NOT_COMMON, case=False, na=True)
    return common.to_numpy()


def eligible(activity, min_events=300_000, min_delta_t_ms=180.0, numerator_ms=2.34e7):
    """Rows of a pilot activity table that pass the selection rule; adds a delta_t_ms column."""
    table = activity.copy()
    table["delta_t_ms"] = numerator_ms / table["price_changes"]
    rule = table["stock"] & (table["book_events"] >= min_events) & (table["delta_t_ms"] >= min_delta_t_ms)
    return table[rule.fillna(False).astype(bool)].sort_values("tick_size_frac")
