from __future__ import annotations

from pathlib import Path
import shutil

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "tracking": "nicolemachado/transportation-and-logistics-tracking-dataset",
    "operations": "yogape/logistics-operations-database",
    "supply_chain": "datasetengineer/logistics-and-supply-chain-dataset",
    "us_performance": "shahriarkabir/us-logistics-performance-dataset",
}

for name, slug in DATASETS.items():
    print(f"Downloading {slug} ...")
    src = Path(kagglehub.dataset_download(slug))
    dst = RAW / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(" ->", dst)

print("All four Kaggle datasets are available under", RAW)
