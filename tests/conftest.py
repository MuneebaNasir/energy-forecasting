import glob
import os
import sys
import time

import pytest

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ["TZ"] = "UTC"
time.tzset()

if "JAVA_HOME" not in os.environ:
    jdks = sorted(glob.glob(os.path.expanduser("~/.jdk/jdk-17*")))
    if jdks:
        home = jdks[-1] + "/Contents/Home"  # macOS JDK layout
        os.environ["JAVA_HOME"] = home if os.path.isdir(home) else jdks[-1]


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    s = (SparkSession.builder.master("local[2]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    yield s
    s.stop()
