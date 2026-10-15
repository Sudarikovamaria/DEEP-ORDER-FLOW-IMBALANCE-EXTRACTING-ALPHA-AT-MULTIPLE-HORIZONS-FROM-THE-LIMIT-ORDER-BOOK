"""Download IEX DEEP days unattended: retries with resume, saved file listing, progress log.

    python scripts/download_days.py --save-listing          # while iextrading.com is reachable
    caffeinate -i nohup .venv/bin/python -u scripts/download_days.py 20261013 > data/raw/download.log 2>&1 &

Downloads only need storage.googleapis.com once the listing is saved, so they survive a change of
network (VPN on/off): a dropped connection is retried and the file resumes from its last byte.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from deepofi.iex import hist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LISTING = PROJECT_ROOT / "data" / "meta" / "hist_listing_deep.json"


def log(message):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dates", nargs="*", help="YYYYMMDD")
    parser.add_argument("--save-listing", action="store_true", help="fetch and save the HIST listing, then exit")
    parser.add_argument("--dest", type=Path, default=hist.RAW_DIR)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--log-every", type=float, default=30.0, help="seconds between progress lines")
    return parser.parse_args()


class Progress:
    def __init__(self, every):
        self.every = every
        self.last_time = time.time()
        self.last_bytes = None

    def __call__(self, done, total):
        now = time.time()
        if self.last_bytes is None:
            self.last_bytes = done
        if now - self.last_time >= self.every:
            speed = (done - self.last_bytes) / (now - self.last_time) / 1e6
            log(f"  {done / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * done / total:5.1f}%), {speed:5.1f} MB/s")
            self.last_time, self.last_bytes = now, done


def download_with_retries(entry, dest, max_attempts, log_every):
    done = Path(dest) / f"{entry['date']}_IEXTP1_{entry['feed']}{entry['version']}.pcap.gz"
    if done.exists() and done.stat().st_size == entry["size"]:
        log(f"  already downloaded: {done.name}")
        return done
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        part = Path(dest) / f"{entry['date']}_IEXTP1_{entry['feed']}{entry['version']}.pcap.gz.part"
        before = part.stat().st_size if part.exists() else 0
        try:
            path = hist.download(entry, dest_dir=dest, progress=Progress(log_every))
            got = entry["size"] - before
            log(f"  done {path.name}: {got / 1e9:.2f} GB in {(time.time() - started) / 60:.1f} min, "
                f"{got / 1e6 / max(time.time() - started, 1e-9):.1f} MB/s average")
            return path
        except Exception as error:  # network errors of every kind: retry and resume
            wait = min(60, 5 * attempt)
            log(f"  attempt {attempt} failed: {type(error).__name__}: {error}; retry in {wait} s")
            time.sleep(wait)
    raise SystemExit(f"Giving up on {entry['date']} after {max_attempts} attempts")


def main():
    args = parse_args()
    if args.save_listing:
        files = hist.list_files()
        LISTING.parent.mkdir(parents=True, exist_ok=True)
        LISTING.write_text(json.dumps(files))
        log(f"Saved listing: {len(files)} days, {min(files)} to {max(files)} -> {LISTING}")
        return

    files = json.loads(LISTING.read_text())
    missing = [d for d in args.dates if d not in files]
    if missing:
        raise SystemExit(f"Not in the saved listing (holiday or not published yet): {missing}")

    for date in args.dates:
        log(f"{date}: {files[date]['size'] / 1e9:.2f} GB")
        download_with_retries(files[date], args.dest, args.max_attempts, args.log_every)
    log("All downloads finished")


if __name__ == "__main__":
    main()
