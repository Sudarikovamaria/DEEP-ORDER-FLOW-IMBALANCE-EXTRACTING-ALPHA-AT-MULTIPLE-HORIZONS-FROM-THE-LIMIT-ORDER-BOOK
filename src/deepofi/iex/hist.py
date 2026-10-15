"""IEX HIST file listing and download.

Using HIST data requires accepting the IEX Historical Data Terms of Use. Raw pcap files are
downloaded to data/raw/, parsed, and deleted; they must never be committed.
"""
import json
import urllib.request
from pathlib import Path

HIST_URL = "https://iextrading.com/api/1.0/hist"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def list_files(feed="DEEP", version="1.0", url=HIST_URL):
    """Return {YYYYMMDD: {"link", "size", ...}} for one feed."""
    with urllib.request.urlopen(url, timeout=60) as response:
        listing = json.load(response)

    files = {}
    for date, entries in listing.items():
        for entry in entries if isinstance(entries, list) else [entries]:
            if entry["feed"] == feed and entry["version"] == version:
                files[date] = {**entry, "size": int(entry["size"])}
    return dict(sorted(files.items()))


def download(entry, dest_dir=RAW_DIR, chunk_bytes=1 << 22, progress=None):
    """Download one HIST file, resuming a partial download. Returns the local path.

    `progress(done_bytes, total_bytes)` is called after every chunk."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{entry['date']}_IEXTP1_{entry['feed']}{entry['version']}.pcap.gz"
    partial = path.with_name(path.name + ".part")

    if path.exists() and path.stat().st_size == entry["size"]:
        return path

    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(entry["link"], headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            offset = 0
        with partial.open("ab" if offset else "wb") as file:
            done = offset
            while chunk := response.read(chunk_bytes):
                file.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, entry["size"])

    if partial.stat().st_size != entry["size"]:
        raise OSError(f"Incomplete download {partial}: {partial.stat().st_size} of {entry['size']} bytes")
    partial.rename(path)
    return path
