"""Run the bronze -> silver -> gold pipeline locally with the same transforms as the
Databricks notebooks, writing parquet to data/local/ instead of Delta tables.

    python scripts/run_pipeline_local.py
"""
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
if "JAVA_HOME" not in os.environ:
    jdk = sorted(glob.glob(os.path.expanduser("~/.jdk/jdk-17*")))[-1]
    os.environ["JAVA_HOME"] = jdk + "/Contents/Home" if os.path.isdir(jdk + "/Contents/Home") else jdk

from pyspark.sql import SparkSession  # noqa: E402

from energyfc import cleaning, config, features  # noqa: E402

RAW, OUT = ROOT / "data/raw", ROOT / "data/local"
START, END = "2016-01-01 00:00:00", "2017-12-31 23:00:00"

spark = (SparkSession.builder.master("local[*]").appName("energyfc-local")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

wide = spark.read.csv(str(RAW / "electricity.csv"), header=True)
weather_raw = spark.read.csv(str(RAW / "weather.csv"), header=True)
metadata = spark.read.csv(str(RAW / "metadata.csv"), header=True)

bronze = cleaning.melt_meters(wide, config.SITES)
bronze.write.mode("overwrite").parquet(str(OUT / "bronze_meters"))
bronze = spark.read.parquet(str(OUT / "bronze_meters"))
print("bronze rows:", bronze.count())

silver = cleaning.clean_meters(bronze, START, END)
silver.write.mode("overwrite").parquet(str(OUT / "silver_meters"))
silver = spark.read.parquet(str(OUT / "silver_meters"))
quality = cleaning.meter_quality_report(silver)
quality.write.mode("overwrite").parquet(str(OUT / "silver_meter_quality"))
quality = spark.read.parquet(str(OUT / "silver_meter_quality"))
weather = cleaning.clean_weather(weather_raw, config.SITES, START, END)
weather.write.mode("overwrite").parquet(str(OUT / "silver_weather"))
weather = spark.read.parquet(str(OUT / "silver_weather"))

print("flags:")
silver.groupBy("quality_flag").count().orderBy("count", ascending=False).show()
print("meters kept:", quality.where("keep").count(), "/", quality.count())

gold = features.build_gold(silver, quality, weather, metadata)
gold.write.mode("overwrite").parquet(str(OUT / "gold_features"))
print("gold rows:", spark.read.parquet(str(OUT / "gold_features")).count())
spark.stop()
