"""Minimal stand-in for the Dataiku Python API, used to run dataiku/recipes/*.py locally.

Datasets are gzipped CSVs in $DATAIKU_STUB_DIR, which is also the type round-trip a
Dataiku *uploaded files* dataset goes through (booleans/dates arrive as strings).
"""
import json
import os
from pathlib import Path

import pandas as pd

_DIR = Path(os.environ.get("DATAIKU_STUB_DIR", "."))


class Dataset:
    def __init__(self, name):
        self.path = _DIR / f"{name}.csv.gz"

    def get_dataframe(self, **kwargs):
        return pd.read_csv(self.path, low_memory=False)

    def write_with_schema(self, df):
        df.to_csv(self.path, index=False)
        print(f"  wrote {self.path.name}: {len(df):,} rows x {df.shape[1]} cols")


def get_custom_variables():
    return json.loads(os.environ.get("DATAIKU_STUB_VARS", "{}"))
