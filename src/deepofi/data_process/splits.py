from datetime import datetime, timedelta

WINDOW_PERIODS = [
    {"val": ("20260720", "20260724"), "train": ("20260727", "20260821"), "test": ("20260824", "20260828")},
    {"val": ("20260810", "20260814"), "train": ("20260817", "20260911"), "test": ("20260914", "20260918")},
    {"val": ("20260831", "20260904"), "train": ("20260908", "20261002"), "test": ("20261005", "20261009")},
]
HOLIDAYS = {"20260907"}


def trading_dates(start, end):
    day = datetime.strptime(start, "%Y%m%d").date()
    last_day = datetime.strptime(end, "%Y%m%d").date()
    dates = []

    while day <= last_day:
        value = day.strftime("%Y%m%d")
        if day.weekday() < 5 and value not in HOLIDAYS:
            dates.append(value)
        day += timedelta(days=1)

    return dates


def get_splits():
    windows = []

    for number, periods in enumerate(WINDOW_PERIODS, start=1):
        window = {"name": f"window_{number:02d}"}
        for part, (start, end) in periods.items():
            window[f"{part}_dates"] = trading_dates(start, end)
        windows.append(window)

    return windows