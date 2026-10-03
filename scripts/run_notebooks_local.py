"""Smoke-test the Databricks notebooks locally: execute 01-04 as written, with a local
Spark session, a minimal `dbutils`/`display` stand-in and the `spark_catalog` metastore
instead of Unity Catalog. 00 (volume download) and 05 (volume export) are Databricks-only.

    python scripts/run_notebooks_local.py
"""
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
if "JAVA_HOME" not in os.environ:
    jdk = sorted(glob.glob(os.path.expanduser("~/.jdk/jdk-17*")))[-1]
    os.environ["JAVA_HOME"] = jdk + "/Contents/Home" if os.path.isdir(jdk + "/Contents/Home") else jdk

from pyspark.sql import DataFrame, SparkSession  # noqa: E402

WAREHOUSE = ROOT / "data/local/warehouse"
OVERRIDES = {"catalog": "spark_catalog", "schema": "energy", "raw_dir": str(ROOT / "data/raw")}
NOTEBOOKS = ["01_bronze_ingest", "02_silver_clean", "03_gold_features", "04_eda"]


class _Widgets:
    def __init__(self):
        self.values = {}

    def text(self, name, default, label=None):
        self.values[name] = OVERRIDES.get(name, default)

    def get(self, name):
        return self.values[name]


class _DBUtils:
    widgets = _Widgets()


def display(obj):
    if isinstance(obj, DataFrame):
        obj = obj.limit(8).toPandas()
    print(obj.head(8).to_string() if hasattr(obj, "head") else obj)


spark = (SparkSession.builder.master("local[*]").appName("notebooks-local")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.warehouse.dir", str(WAREHOUSE)).getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.sql("CREATE DATABASE IF NOT EXISTS energy")

os.chdir(ROOT / "databricks")  # notebooks resolve ../src relative to their own folder
for name in NOTEBOOKS:
    print(f"\n===== {name} =====")
    src = (ROOT / "databricks" / f"{name}.py").read_text()
    exec(compile(src, f"{name}.py", "exec"), {"spark": spark, "dbutils": _DBUtils(), "display": display,
                                               "__name__": "__main__"})
print("\nall notebooks ran")
