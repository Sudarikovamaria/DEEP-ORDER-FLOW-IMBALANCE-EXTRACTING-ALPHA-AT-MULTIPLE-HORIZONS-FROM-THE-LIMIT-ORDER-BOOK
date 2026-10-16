"""Write or verify the dataset manifest: SHA-256 of every book and meta file of configs/tickers.yaml.

    python scripts/freeze_dataset.py                          # write data/MANIFEST.json
    python scripts/freeze_dataset.py --verify --root /path/to/deepofi_dataset

The manifest also records the dataset version and the pipeline config it was built with, so a
copy received from a teammate can be checked before training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

from deepofi.data_process.splits import get_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--tickers", type=Path, default=PROJECT_ROOT / "configs" / "tickers.yaml")
    parser.add_argument("--pipeline", type=Path, default=PROJECT_ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_files(universe):
    tickers = [t for group in universe["tickers"].values() for t in group]
    dates = sorted({d for w in get_splits() for part in ("val_dates", "train_dates", "test_dates") for d in w[part]})
    return [f"{folder}/{t}/{d}.{ext}" for t in tickers for d in dates for folder, ext in (("book", "parquet"), ("meta", "json"))]


def main():
    args = parse_args()
    universe = yaml.safe_load(args.tickers.read_text())
    files = dataset_files(universe)
    manifest_path = args.root / "MANIFEST.json"

    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        bad = [f for f, digest in manifest["files"].items() if not (args.root / f).exists() or sha256(args.root / f) != digest]
        print(f"Dataset {manifest['dataset_version']}: {len(manifest['files']) - len(bad)}/{len(manifest['files'])} files match")
        if bad:
            raise SystemExit(f"Missing or changed: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
        return

    missing = [f for f in files if not (args.root / f).exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing[:10]}")
    manifest = {
        "dataset_version": universe["dataset_version"],
        "tickers": universe["tickers"],
        "period": universe["period"],
        "pipeline_config": yaml.safe_load(args.pipeline.read_text()),
        "files": {f: sha256(args.root / f) for f in files},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {manifest_path}: dataset {manifest['dataset_version']}, {len(files)} files")


if __name__ == "__main__":
    main()
