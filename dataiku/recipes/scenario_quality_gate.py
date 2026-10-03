# Dataiku scenario step (Execute Python code), runs after "Build metrics_overall".
# Fails the scenario, which triggers the e-mail/Slack reporter, if the retrained
# model stops beating the seasonal-naive baseline: a silent regression is worse than a loud one.
from dataiku.scenario import Scenario

import dataiku

overall = dataiku.Dataset("metrics_overall").get_dataframe().set_index("model")
lgbm, snaive = overall.loc["lgbm", "median_mase"], overall.loc["snaive_168", "median_mase"]
msg = f"median MASE lgbm={lgbm:.3f} vs seasonal naive={snaive:.3f}"
Scenario().set_scenario_variables(last_quality_gate=msg)
if lgbm >= 0.95 * snaive:
    raise Exception("Quality gate failed: " + msg)
print("Quality gate passed: " + msg)
