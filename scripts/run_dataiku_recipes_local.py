"""Run the Dataiku recipe scripts unchanged against the CSV exports, through the
stand-in `dataiku` module in tests/dataiku_stub. Catches type/IO issues before the
code is pasted into Dataiku.

    python scripts/export_for_dataiku.py
    python scripts/run_dataiku_recipes_local.py [--folds 2]
"""
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/dataiku_local"
WORK.mkdir(parents=True, exist_ok=True)
for f in (ROOT / "data/exports").glob("*.csv.gz"):
    shutil.copy(f, WORK / f.name)

folds = sys.argv[sys.argv.index("--folds") + 1] if "--folds" in sys.argv else "12"
os.environ["DATAIKU_STUB_DIR"] = str(WORK)
os.environ["DATAIKU_STUB_VARS"] = json.dumps({"test_start": "2017-01-01", "n_folds": folds})
sys.path[:0] = [str(ROOT / "tests/dataiku_stub"), str(ROOT / "src")]

for recipe in ["compute_backtest", "compute_dm_tests", "compute_anomalies", "compute_intervals",
               "scenario_quality_gate"]:
    print(f"== {recipe}")
    runpy.run_path(str(ROOT / f"dataiku/recipes/{recipe}.py"), run_name="__main__")
print("all recipes ran")
