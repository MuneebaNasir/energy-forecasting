"""Zip the pandas-only part of energyfc for upload to the Dataiku project library."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "config", "metrics", "stats_tests", "backtest", "ensembles", "anomalies", "signature"]
out = ROOT / "dataiku/project_lib/energyfc.zip"
with zipfile.ZipFile(out, "w") as z:
    for m in MODULES:
        z.write(ROOT / f"src/energyfc/{m}.py", f"energyfc/{m}.py")
print("wrote", out)
