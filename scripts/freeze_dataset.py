"""Write or verify the dataset manifest: SHA-256 of every book and meta file of configs/tickers.yaml.

    python scripts/freeze_dataset.py                          # write data/MANIFEST.json
    python scripts/freeze_dataset.py --verify --root /path/to/deepofi_dataset
    python scripts/freeze_dataset.py --export data/export/deepofi_dataset_v1.tar

The manifest also records the dataset version and the pipeline config it was built with, so a
copy received from a teammate can be checked before training.
"""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import yaml

from deepofi.data_process.splits import get_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = """deepofi_dataset_v1
==================

Очищенные стаканы (10 уровней) IEX DEEP для 11 акций за 59 торговых дней,
20 июля – 9 октября 2026 года. Собрано scripts/build_dataset.py, параметры —
configs/pipeline.yaml (копия внутри MANIFEST.json).

Тикеры: DIS, CTSH, UBER (small tick); MO, NEE, OXY, USB (средний); BSX, NKE, VZ, BAC (large tick).

Структура:
  book/{TICKER}/{YYYYMMDD}.parquet   очищенный стакан, одна строка = одно событие
  meta/{TICKER}/{YYYYMMDD}.json      характеристики дня (число событий, изменений mid, tick size, ...)
  MANIFEST.json                      версия датасета, конфиг и SHA-256 каждого файла

Как использовать:
  1. Распакуйте архив в любую папку вне репозитория.
  2. Проверьте копию из корня репозитория:
       python scripts/freeze_dataset.py --verify --root /путь/deepofi_dataset_v1
  3. Укажите эту папку при обучении:
       python -m deepofi.cli train --model lstm --ticker BAC --input of --data-root /путь/deepofi_dataset_v1

Признаки (OF, LOB, OFI) и таргеты считаются из стаканов на лету; датасет не менять.
Нашли проблему — выпускаем v2, а не правим v1.

Атрибуция (обязательна по условиям IEX):
  Data provided for free by IEX.
  IEX Historical Data Terms of Use: https://www.iex.io/products/market-data-connectivity/hist-terms
Не публикуйте данные открыто и не коммитьте их в репозиторий.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--tickers", type=Path, default=PROJECT_ROOT / "configs" / "tickers.yaml")
    parser.add_argument("--pipeline", type=Path, default=PROJECT_ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--export", type=Path, help="write a tar with the manifest files, MANIFEST.json and README.txt")
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


def export(root, manifest_path, out_path):
    """Plain tar (no compression, parquet is already compressed) under one top folder named after the version."""
    manifest = json.loads(manifest_path.read_text())
    top = f"deepofi_dataset_{manifest['dataset_version']}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        readme = README.encode("utf-8")
        info = tarfile.TarInfo(f"{top}/README.txt")
        info.size = len(readme)
        archive.addfile(info, io.BytesIO(readme))
        archive.add(manifest_path, arcname=f"{top}/MANIFEST.json")
        for name in manifest["files"]:
            archive.add(root / name, arcname=f"{top}/{name}")
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e9:.2f} GB, {len(manifest['files']) + 2} files), sha256 {sha256(out_path)}")


def main():
    args = parse_args()
    universe = yaml.safe_load(args.tickers.read_text())
    files = dataset_files(universe)
    manifest_path = args.root / "MANIFEST.json"

    if args.export:
        export(args.root, manifest_path, args.export)
        return

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
