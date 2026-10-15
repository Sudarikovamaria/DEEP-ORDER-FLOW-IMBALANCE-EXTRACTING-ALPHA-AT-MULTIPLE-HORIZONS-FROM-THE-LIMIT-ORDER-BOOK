"""Unattended dataset download: selection-day pilot, then download -> extract candidates -> delete pcap.

    caffeinate -i nohup .venv/bin/python -u scripts/overnight.py > data/overnight.log 2>&1 &

1. Selection day (before the period): download, run the pilot on all common stocks, save the
   candidate list (common stocks with >= --min-events book events) to data/meta/candidates.json.
2. Every day (selection day included): download, extract the candidates' records to
   data/records/{date}.bin in a worker process while the next day downloads, delete the pcap.

Safe to rerun: finished days are skipped and partial downloads resume. The HIST listing and the
symbol directory are fetched on the first run if missing; after that only storage.googleapis.com
is needed.
"""
import argparse
import fcntl
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from download_days import download_with_retries, load_listing, log
from pilot_day import run_pilot

from deepofi.iex import hist
from deepofi.iex.deep import extract_day

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META = PROJECT_ROOT / "data" / "meta"
CANDIDATES = META / "candidates.json"
LOCK = PROJECT_ROOT / "data" / "overnight.lock"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-day", default="20260717")
    parser.add_argument("--start", default="20260720")
    parser.add_argument("--end", default="20261009")
    parser.add_argument("--min-events", type=int, default=200_000)
    parser.add_argument("--records", type=Path, default=PROJECT_ROOT / "data" / "records")
    parser.add_argument("--raw", type=Path, default=hist.RAW_DIR)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--keep-pcap", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    return parser.parse_args()


def check_disk(path, need_bytes, min_free_gb):
    free = shutil.disk_usage(path).free
    if free < need_bytes + min_free_gb * 1e9:
        raise SystemExit(f"Not enough disk space: {free / 1e9:.0f} GB free, need {need_bytes / 1e9:.0f} GB + {min_free_gb:.0f} GB reserve")


def finish(pending, keep_pcap):
    future, date, pcap, out, started = pending
    try:
        summary = future.result()
    except Exception as error:
        log(f"{date}: extraction FAILED ({type(error).__name__}: {error}); pcap kept at {pcap}")
        return
    summary.update({"date": date, "seconds": round(time.time() - started), "gb": round(out.stat().st_size / 1e9, 2)})
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    log(f"{date}: extracted {summary['records']:,} records ({summary['gb']} GB) in {summary['seconds']} s, "
        f"gaps {summary['gap_messages']}, duplicates {summary['duplicate_packets']}, truncated bytes {summary['truncated_bytes']}")
    if not keep_pcap:
        Path(pcap).unlink()
        log(f"{date}: deleted {Path(pcap).name}")


def acquire_lock():
    """Exit if another copy is running: two copies would append to the same partial downloads."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another overnight.py is already running; stop it first (pgrep -fl overnight.py)")
    return handle


def main():
    args = parse_args()
    lock = acquire_lock()  # noqa: F841 - held until the process exits
    files = load_listing()
    period = sorted(d for d in files if args.start <= d <= args.end)
    days = [args.selection_day] + period
    log(f"Selection day {args.selection_day}, period {period[0]}-{period[-1]}: {len(period)} trading days, "
        f"{sum(files[d]['size'] for d in days) / 1e9:.0f} GB to download")

    if not args.candidates.exists():
        entry = files[args.selection_day]
        check_disk(args.raw, entry["size"], args.min_free_gb)
        log(f"{args.selection_day}: downloading selection day ({entry['size'] / 1e9:.1f} GB)")
        pcap = download_with_retries(entry, args.raw, max_attempts=200, log_every=60)
        _, symbols = run_pilot(pcap, META, stocks_only=True, min_events=args.min_events, log=log)
        args.candidates.write_text(json.dumps({"selection_day": args.selection_day, "min_events": args.min_events, "symbols": symbols}, indent=2))
        log(f"Saved {len(symbols)} candidates to {args.candidates}")
    symbols = json.loads(args.candidates.read_text())["symbols"]

    pending = None
    with ProcessPoolExecutor(max_workers=1) as pool:
        for i, date in enumerate(days, start=1):
            out = args.records / f"{date}.bin"
            if out.exists():
                log(f"{date}: records already extracted, skipping ({i}/{len(days)})")
                continue
            entry = files[date]
            check_disk(args.raw, 2 * entry["size"], args.min_free_gb)
            log(f"{date}: downloading ({i}/{len(days)}, {entry['size'] / 1e9:.1f} GB)")
            pcap = download_with_retries(entry, args.raw, max_attempts=200, log_every=60)
            if pending:
                finish(pending, args.keep_pcap)
            pending = (pool.submit(extract_day, str(pcap), str(out), symbols), date, pcap, out, time.time())
        if pending:
            finish(pending, args.keep_pcap)

    done = sum((args.records / f"{d}.bin").exists() for d in days)
    log(f"Finished: {done}/{len(days)} days extracted to {args.records}")


if __name__ == "__main__":
    main()
