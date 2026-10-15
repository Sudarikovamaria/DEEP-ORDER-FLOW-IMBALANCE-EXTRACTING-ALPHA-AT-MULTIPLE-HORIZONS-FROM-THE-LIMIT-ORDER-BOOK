import pandas as pd

from deepofi.data_process.universe import eligible, is_common_stock, load_symbol_directory

DIRECTORY = """Nasdaq Traded|Symbol|Security Name|Listing Exchange|Market Category|ETF|Round Lot Size|Test Issue|Financial Status|CQS Symbol|NASDAQ Symbol|NextShares
Y|BAC|Bank of America Corporation Common Stock|N| |N|100|N||BAC|BAC|N
Y|EQNR|Equinor ASA|N| |N|100|N||EQNR|EQNR|N
Y|SPY|SPDR S&P 500 ETF Trust|P| |Y|100|N||SPY|SPY|N
Y|O|Realty Income Corporation Common Stock|N| |N|100|N||O|O|N
Y|PSEC|Prospect Capital Corp 5.35% Series A Fixed Rate Cumulative Perpetual Preferred Stock|Q|Q|N|100|N||PSEC|PSEC|N
Y|ZVZZT|NASDAQ TEST STOCK|Q|G|N|100|Y||ZVZZT|ZVZZT|N
File Creation Time: 1015202606:01|||||||||||
"""


def test_common_stock_filter(tmp_path):
    path = tmp_path / "nasdaqtraded.txt"
    path.write_text(DIRECTORY)
    directory = load_symbol_directory(path)

    assert "ZVZZT" not in directory.index
    assert is_common_stock(["BAC", "EQNR", "SPY", "O", "PSEC", "UNKNOWN"], directory).tolist() == [True, True, False, True, False, False]


def test_eligible_rule():
    activity = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "stock": [True, True, True, False],
        "book_events": [400_000, 400_000, 200_000, 900_000],
        "price_changes": [100_000.0, 200_000.0, 50_000.0, 50_000.0],  # delta t 234, 117, 468, 468 ms
        "tick_size_frac": [0.9, 0.1, 0.5, 0.9],
    })
    assert eligible(activity)["symbol"].tolist() == ["A"]
    assert eligible(activity, min_events=100_000)["symbol"].tolist() == ["C", "A"]
