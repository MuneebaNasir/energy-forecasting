"""Export the local gold/silver tables as gzipped CSVs ready to upload to Dataiku
(the Free Edition path, when there is no Databricks connection).

    python scripts/export_for_dataiku.py      # -> data/exports/*.csv.gz
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data/local", ROOT / "data/exports"
OUT.mkdir(parents=True, exist_ok=True)

for table in ["gold_features", "silver_meters", "silver_meter_quality"]:
    df = pd.read_parquet(SRC / table)
    df.to_csv(OUT / f"{table}.csv.gz", index=False)
    print(f"{table}: {len(df):,} rows -> {(OUT / f'{table}.csv.gz').stat().st_size / 1e6:.0f} MB")
sig = SRC / "outputs/energy_signatures.csv"
if sig.exists():
    pd.read_csv(sig).to_csv(OUT / "gold_energy_signatures.csv.gz", index=False)
