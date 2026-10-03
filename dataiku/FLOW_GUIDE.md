# Building the Dataiku project

> The project is built automatically by `scripts/build_dataiku_project.py` and exported to
> `dataiku/export/ENERGY_FORECAST.zip`. This guide is the same build, step by step, by hand.

Target Flow (three Flow zones):

```
 ┌─ Zone: Ingestion ──────────────────────────────────────────────┐
 │ gold_features ──[Prepare]──> gold_model_ready ──[Split]──┬─> train_2016
 │ silver_meters                                            └─> test_2017
 │ silver_meter_quality                                           │
 └────────────────────────────────────────────────────────────────┘
 ┌─ Zone: Modeling ───────────────────────────────────────────────┐
 │ train_2016 ──[Visual ML: AutoML]──> ◆ load_forecast (saved model)
 │ test_2017 + ◆ ──[Score]──> test_scored ──[Evaluate]──> ▣ model_evaluations
 │ gold_features ──[Python: compute_backtest]──> predictions, metrics_*, feature_importance
 │ predictions ──[Python: compute_dm_tests]──> dm_tests
 │ gold_features ──[Python: compute_intervals]──> quantile_predictions, interval_scores
 └────────────────────────────────────────────────────────────────┘
 ┌─ Zone: Monitoring ─────────────────────────────────────────────┐
 │ predictions + silver_meters ──[Python: compute_anomalies]──> consumption_anomalies, meter_faults
 │ metrics_per_meter ──[Group by site_id, model]──> metrics_by_site
 └────────────────────────────────────────────────────────────────┘
 Scenario "Monthly retrain"  ·  Dashboard "Energy forecasting"  ·  Wiki "Project handover"
```

## 0. Setup

1. **Project** → New project → *Energy load forecasting* (key `ENERGY_FORECAST`).
2. **Project variables** (⋯ → Variables): `{"test_start": "2017-01-01", "n_folds": "12"}`
3. **Code env** (Administration → Code Envs → New Python env, Python 3.10+):
   packages `pandas>=2 lightgbm>=4 scipy numpy`. Set it as the project default
   (Settings → Code env). *On Dataiku Cloud: Launchpad → Code Envs.*
4. **Library**: run `python scripts/package_dataiku_lib.py`, then **Libraries → python**,
   upload `energyfc.zip` and unzip so you get `python/energyfc/*.py`.

## 1. Datasets (Zone: Ingestion)

- **With a Databricks connection**: + Dataset → Databricks → table `workspace.energy.gold_features`
  (and `silver_meters`, `silver_meter_quality`). No copy: Dataiku pushes SQL down where it can.
- **Without** (Free Edition): run notebook `05_export_for_dataiku` and upload the exported
  CSVs (+ Dataset → Upload your files). Locally, `data/local/*` parquet can be converted with
  `pd.read_parquet(...).to_csv(...)`.

Check the schema: `timestamp` as date, `y`/lags as double, `is_imputed` as boolean.

## 2. Prepare recipe → `gold_model_ready`

Visual steps (each one is something to explain in an interview):
1. **Filter rows**: keep `y` is defined.
2. **Filter rows**: keep `is_imputed` = false (only score/train on real readings).
3. **Formula** `split` = `if(timestamp < parseDate("2017-01-01", "yyyy-MM-dd"), "train", "test")`.
4. Keep `kwh`, `scale` and `quality_flag`: they are needed for evaluation and are *rejected* as
   features in Visual ML (section 4).

Then **Split recipe** on `split` → `train_2016`, `test_2017`.

## 3. Data quality rules on `gold_features`

Dataset → **Data Quality** (or *Status → Checks* on older versions):
- record count > 1,000,000
- `y` empty share < 10%
- `air_temp` min ≥ −30, max ≤ 45
- `building_id` distinct count ≥ 60

## 4. Visual ML (Lab → AutoML Prediction on `train_2016`)

- Target: **`y`** (normalised load). Prediction type: regression.
- **Train/test**: *Explicit extracts*: train = `train_2016`, test = `test_2017`. (A random split
  would leak future information through the lag features: say this out loud in the interview.)
- **Features**: enable exactly the list in `src/energyfc/config.py: FEATURES`; reject
  `building_id`, `timestamp`, `date`, `kwh`, `scale`, `split`, `quality_flag`.
  `primary_use` → dummy encoding; `hour`, `dow`, `month` → numeric (trees handle it).
- **Algorithms**: LightGBM, Random Forest, Ridge regression (linear baseline).
- **Metric**: MAE.
- Train, then compare in the **model comparison** view. Check:
  - variable importance: `y_lag_168`, `y_lag_24`, `hour` should dominate; temperature
    features should appear for heating-dominated buildings;
  - **subpopulation analysis** on `primary_use` and `site_id`: where does it fail?
  - **partial dependence** on `temp_mean_24h`: should rise below ~15 °C (heating).
- **Deploy** the best model to the Flow as `load_forecast`.

## 5. Score + Evaluate

- **Score** recipe: `test_2017` + `load_forecast` → `test_scored`.
- Prepare step on `test_scored`: formula `kwh_hat = prediction * scale`.
- **Evaluate** recipe → Model Evaluation Store `model_evaluations` (gives drift + performance
  tracking across retrains).

## 6. Python recipes (Zone: Modeling / Monitoring)

Create each as *+ Recipe → Python*, set inputs/outputs as in the file header, paste the code from
`dataiku/recipes/`:

| Recipe | Inputs | Outputs |
|---|---|---|
| `compute_backtest.py` | gold_features | predictions, metrics_per_meter, metrics_overall, feature_importance |
| `compute_dm_tests.py` | predictions | dm_tests |
| `compute_anomalies.py` | predictions, silver_meters | consumption_anomalies, meter_faults |
| `compute_intervals.py` | gold_features | quantile_predictions, interval_scores |

Why both Visual ML **and** a Python backtest? Visual ML evaluates on one fixed holdout (all of
2017 with a model trained on 2016 only). The Python recipe retrains monthly (12 rolling origins),
which is how the model would actually run, and adds the significance test Visual ML doesn't do.

## 7. Scenario "Monthly retrain"

- Trigger: **time-based**, monthly, day 1, 03:00 (Europe/Monaco).
- Steps:
  1. Build `gold_features` (only if it's a managed dataset; with Databricks the nightly
     job refreshes it, see `databricks.yml`)
  2. Run checks / data quality on `gold_features`: stop if error
  3. Retrain `load_forecast`
  4. Build `test_scored`, `model_evaluations`, `predictions`, `metrics_overall`, `dm_tests`,
     `consumption_anomalies` (*build required dependencies*)
  5. **Execute Python code**: paste `dataiku/recipes/scenario_quality_gate.py`
- Reporter: mail on failure with `${last_quality_gate}` in the body.

## 8. Dashboard "Energy forecasting"

Charts are built on the datasets (Charts tab), then *Publish → Dashboard*:
1. **Metric tile**: median MASE, LightGBM vs seasonal naive (from `metrics_overall`).
2. **Bar**: `dm_tests`, count by `verdict`, one panel per `model_b`.
3. **Box plot**: `metrics_per_meter`, MASE by `model`.
4. **Line**: `predictions` filtered on one building, `kwh` vs `kwh_hat` for `lgbm` (add a filter
   tile on `building_id`).
5. **Table**: `consumption_anomalies` sorted by |`excess_kwh`|.
6. **Table**: `meter_faults` (hours, fault type) to send to the metering team.
7. **Line + band**: `quantile_predictions` for one building, `q10_cal`/`q90_cal` as an area,
   `q50` and `kwh` as lines; plus a bar of `interval_scores.coverage_calibrated` by month (target 0.80).
8. **Scatter**: `gold_energy_signatures`, `balance_temp_c` vs `heating_pct_per_c`, colour by site.
9. **Text tile** with 3 sentences of findings in French (from `docs/synthese_fr.md`).

## 9. Wiki

Create the wiki home page from `dataiku/WIKI.md`. Then export the project
(⋯ → Export → include datasets: *none*, saved models: yes) and commit the zip to `dataiku/export/`.
