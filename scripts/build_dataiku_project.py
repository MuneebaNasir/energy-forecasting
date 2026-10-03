"""Build the complete Dataiku project ENERGY_FORECAST through the public API.

Creates: project + variables + code env selection, project library (energyfc), uploaded
datasets, flow zones, Prepare/Split visual recipes, Python recipes, Visual ML task with a
deployed saved model, Score + Evaluate recipes, scenario, dashboard, wiki; then runs the
Flow and exports the project to dataiku/export/.

    export DSS_URL=http://localhost:11000 DSS_API_KEY=...   # or ~/dataiku/.api_key
    python scripts/build_dataiku_project.py [--recreate] [--stages setup,flow,ml,run,ops,dashboard,export]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import dataikuapi
import dataikuapi.utils

ROOT = Path(__file__).resolve().parents[1]
PROJECT_KEY = "ENERGY_FORECAST"
CODE_ENV = "energyfc"
CONN = "filesystem_managed"
LIB_MODULES = ["__init__", "config", "metrics", "stats_tests", "backtest", "ensembles", "anomalies",
               "signature", "deep", "plots"]
UPLOADS = ["gold_features", "silver_meters", "silver_meter_quality", "gold_energy_signatures"]
STRING_COLS = {"building_id", "site_id", "timestamp", "date", "quality_flag", "primary_use"}
INT_COLS = {"hour", "dow", "month", "is_weekend", "is_holiday", "n_days"}
BOOL_COLS = {"is_imputed", "keep"}


def column_type(name):
    if name in STRING_COLS:
        return "string"
    if name in INT_COLS:
        return "bigint"
    if name in BOOL_COLS:
        return "boolean"
    return "double"


def set_types(ds):
    """CSV uploads are detected as all-string; give every column its real storage type."""
    settings = ds.get_settings()
    for col in settings.get_raw()["schema"]["columns"]:
        col["type"] = column_type(col["name"])
    settings.save()

# name -> (zone, inputs, outputs)
PY_RECIPES = {
    "compute_backtest": ("Modeling", ["gold_features"],
                         ["predictions", "metrics_per_meter", "metrics_overall", "feature_importance"]),
    "compute_gru": ("Modeling", ["gold_features"], ["gru_predictions"]),
    "compute_ensemble": ("Modeling", ["predictions", "gru_predictions", "gold_features"],
                         ["predictions_all", "metrics_all", "champion_per_meter"]),
    "compute_dm_tests": ("Modeling", ["predictions_all"], ["dm_tests"]),
    "compute_intervals": ("Modeling", ["gold_features"], ["quantile_predictions", "interval_scores"]),
    "compute_anomalies": ("Monitoring", ["predictions", "silver_meters"],
                          ["consumption_anomalies", "meter_faults"]),
    "publish_insights": ("Monitoring", ["predictions_all", "metrics_all", "quantile_predictions",
                                        "consumption_anomalies", "gold_features"], ["insights_log"]),
}


def client():
    key = os.environ.get("DSS_API_KEY") or Path("~/dataiku/.api_key").expanduser().read_text().strip()
    return dataikuapi.DSSClient(os.environ.get("DSS_URL", "http://localhost:11000"), key)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- setup
def stage_setup(c, recreate):
    if PROJECT_KEY in c.list_project_keys():
        if not recreate:
            log("project exists (use --recreate to rebuild from scratch)")
            return c.get_project(PROJECT_KEY)
        c.get_project(PROJECT_KEY).delete(clear_managed_datasets=True)
    p = c.create_project(PROJECT_KEY, "Energy load forecasting", "admin",
                         description="Day-ahead meter-level electricity forecasting, meter-fault and "
                                     "consumption-anomaly detection (BDG2, 76 meters).")
    v = p.get_variables()
    v["standard"] = {"test_start": "2017-01-01", "n_folds": "12"}
    p.set_variables(v)

    s = p.get_settings()
    s.get_raw()["settings"]["codeEnvs"]["python"] = {"mode": "EXPLICIT_ENV", "envName": CODE_ENV}
    s.save()

    lib = p.get_library()
    py = lib.get_folder("/python")
    folder = py.get_child("energyfc") or py.add_folder("energyfc")
    for m in LIB_MODULES:
        f = folder.get_child(f"{m}.py") or folder.add_file(f"{m}.py")
        f.write((ROOT / f"src/energyfc/{m}.py").read_text())
    log(f"library: python/energyfc ({len(LIB_MODULES)} modules)")

    flow = p.get_flow()
    zones = {z.name: z for z in flow.list_zones()}
    for name, color in [("Ingestion", "#2ab1ac"), ("Modeling", "#2a78d6"), ("Monitoring", "#eb6834")]:
        if name not in zones:
            zones[name] = flow.create_zone(name, color)

    for name in UPLOADS:
        path = ROOT / f"data/exports/{name}.csv.gz"
        ds = p.create_upload_dataset(name)
        with open(path, "rb") as fh:
            ds.uploaded_add_file(fh, path.name)
        ds.autodetect_settings().save()
        set_types(ds)
        ds.move_to_zone(zones["Ingestion"])
        log(f"uploaded {name} ({path.stat().st_size / 1e6:.0f} MB)")
    return p


# ---------------------------------------------------------------- flow
def zone(p, name):
    return next(z for z in p.get_flow().list_zones() if z.name == name)


def ensure_visual_recipes(p):
    existing = {r["name"] for r in p.list_recipes(as_type="listitems")}
    if "prepare_model_ready" not in existing:
        b = p.new_recipe("prepare", "prepare_model_ready")
        b.with_input("gold_features")
        b.with_new_output("gold_model_ready", CONN)
        r = b.create()
        st = r.get_settings()
        st.add_processor_step("RemoveRowsOnEmpty", {"columns": ["y"], "keep": False, "appliesTo": "SINGLE_COLUMN"})
        st.add_processor_step("FilterOnValue", {
            "appliesTo": "SINGLE_COLUMN", "columns": ["is_imputed"], "action": "REMOVE_ROW",
            "values": ["true"], "matchingMode": "FULL_STRING", "normalizationMode": "LOWERCASE"})
        st.add_processor_step("CreateColumnWithGREL", {
            "column": "split", "expression": 'if(timestamp < "2017-01-01", "train", "test")'})
        st.save()
        r.compute_schema_updates().apply()
    if "split_train_test" not in existing:
        b = p.new_recipe("split", "split_train_test")
        b.with_input("gold_model_ready")
        b.with_new_output("train_2016", CONN)
        b.with_new_output("test_2017", CONN)
        st = b.create().get_settings()
        pl = st.get_json_payload()
        pl.update({"mode": "VALUES", "column": "split", "defaultOutputIndex": -1,
                   "valueSplits": [{"value": "train", "outputIndex": 0}, {"value": "test", "outputIndex": 1}]})
        st.set_json_payload(pl)
        st.save()
    for name in ["gold_model_ready", "train_2016", "test_2017"]:
        p.get_dataset(name).move_to_zone(zone(p, "Ingestion"))


def ensure_python_recipes(p):
    existing = {r["name"] for r in p.list_recipes(as_type="listitems")}
    datasets = {d["name"] for d in p.list_datasets()}
    for name, (zone_name, inputs, outputs) in PY_RECIPES.items():
        code = (ROOT / f"dataiku/recipes/{name}.py").read_text()
        if name in existing:
            st = p.get_recipe(name).get_settings()
            st.set_code(code)
            st.save()
            continue
        b = p.new_recipe("python", name)
        for i in inputs:
            b.with_input(i)
        for o in outputs:
            if o in datasets:
                b.with_output(o)
            else:
                b.with_new_output_dataset(o, CONN)
                datasets.add(o)
        b.with_script(code)
        b.create()
        for o in outputs:
            p.get_dataset(o).move_to_zone(zone(p, zone_name))
        log(f"recipe {name}: {inputs} -> {outputs}")


def stage_flow(p):
    ensure_visual_recipes(p)
    ensure_python_recipes(p)


# ---------------------------------------------------------------- run
BUILD_ORDER = ["train_2016", "test_2017", "metrics_overall", "gru_predictions", "metrics_all", "dm_tests",
               "interval_scores", "consumption_anomalies", "insights_log"]


def stage_run(p):
    """Build every Flow output in dependency order (each build also builds missing upstream)."""
    for name in BUILD_ORDER:
        t0 = time.time()
        p.get_dataset(name).build(job_type="RECURSIVE_BUILD")
        log(f"built {name} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- ml: Visual ML + score + evaluate
LGBM_HP = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
           "subsample": 0.8, "colsample_bytree": 0.8}


def stage_ml(p):
    sys.path.insert(0, str(ROOT / "src"))
    from energyfc.config import FEATURES

    if any(m["name"] == "load_forecast" for m in p.list_saved_models()):
        log("saved model load_forecast exists, skipping")
        return
    for d in ["train_2016", "test_2017"]:
        p.get_dataset(d).build(job_type="RECURSIVE_BUILD")
    t = p.create_prediction_ml_task("train_2016", "y", ml_backend_type="PY_MEMORY", guess_policy="DEFAULT",
                                    prediction_type="REGRESSION", wait_guess_complete=True)
    st = t.get_settings()
    st.get_split_params().set_split_explicit(
        train_selection={"samplingMethod": "FULL"}, test_selection={"samplingMethod": "FULL"},
        dataset_name="train_2016", test_dataset_name="test_2017")  # never random: lags would leak
    for f in st.get_raw()["preprocessing"]["per_feature"]:
        if f != "y":
            st.use_feature(f) if f in FEATURES else st.reject_feature(f)
    st.get_raw()["preprocessing"]["per_feature"]["primary_use"]["category_handling"] = "DUMMIFY"
    st.disable_all_algorithms()
    st.set_algorithm_enabled("LIGHTGBM_REGRESSION", True)
    st.set_algorithm_enabled("RIDGE_REGRESSION", True)
    lg = st.get_algorithm_settings("LIGHTGBM_REGRESSION")
    for k, v in LGBM_HP.items():
        h = getattr(lg, k)
        h.set_explicit_values([v]) if hasattr(h, "set_explicit_values") else setattr(lg, k, v)
    st.get_raw()["modeling"]["metrics"]["evaluationMetric"] = "MAE"
    st.get_raw()["envSelection"] = {"envMode": "USE_BUILTIN_MODE"}
    st.save()
    t.start_train(session_name="LightGBM vs Ridge, explicit 2016/2017 split")
    t.wait_train_complete()
    algo = {m: t.get_trained_model_details(m).get_raw()["modeling"]["algorithm"] for m in t.get_trained_models_ids()}
    best = next(m for m, a in algo.items() if a == "LIGHTGBM_REGRESSION")
    sm = t.deploy_to_flow(best, "load_forecast", "train_2016", test_dataset="test_2017",
                          redo_optimization=False)["savedModelId"]

    b = p.new_recipe("prediction_scoring", "score_test_2017")
    b.with_input_model(sm)
    b.with_input("test_2017")
    b.with_new_output("test_scored", CONN)
    b.create()
    mes = p.create_model_evaluation_store("model_evaluations")
    b = p.new_recipe("evaluation", "evaluate_load_forecast")
    b.with_input_model(sm)
    b.with_input("test_2017")
    b.with_output_evaluation_store(mes.mes_id)
    b.create()
    p.get_dataset("test_scored").move_to_zone(zone(p, "Modeling"))
    p.get_dataset("test_scored").build(job_type="NON_RECURSIVE_FORCED_BUILD")
    job = p.start_job({"type": "NON_RECURSIVE_FORCED_BUILD", "outputs": [
        {"type": "MODEL_EVALUATION_STORE", "id": mes.mes_id, "projectKey": PROJECT_KEY}]})
    while job.get_status()["baseStatus"]["state"] not in ("DONE", "FAILED", "ABORTED"):
        time.sleep(5)
    perf = t.get_trained_model_details(best).get_performance_metrics()
    log(f"Visual ML: LightGBM deployed as load_forecast (MAE {perf['mae']:.3f}, R2 {perf['r2']:.3f} on 2017)")


# ---------------------------------------------------------------- ops: data quality, scenario, wiki
DQ_RULES = [
    {"type": "RecordCountInRangeRule", "displayName": "At least 1M meter-hours",
     "minimum": 1000000, "minimumEnabled": True, "maximumEnabled": False},
    {"type": "ColumnMinInRangeRule", "displayName": "Temperature >= -30 C", "columns": ["air_temp"],
     "minimum": -30, "minimumEnabled": True, "maximumEnabled": False},
    {"type": "ColumnMaxInRangeRule", "displayName": "Temperature <= 45 C", "columns": ["air_temp"],
     "maximum": 45, "maximumEnabled": True, "minimumEnabled": False},
    {"type": "ColumnNotEmptyRule", "displayName": "Keys never empty", "columns": ["building_id", "timestamp"],
     "thresholdType": "ENTIRE_COLUMN_NOT_EMPTY"},
]
MONITORED = ["test_scored", "metrics_all", "dm_tests", "interval_scores", "consumption_anomalies", "insights_log"]


def ensure_dq_rules(p):
    dq = p.get_dataset("gold_features").get_data_quality_rules()
    have = {r.get_raw().get("displayName") for r in dq.list_rules()}
    for rule in DQ_RULES:
        if rule["displayName"] not in have:
            dq.create_rule({"enabled": True, "autoRun": True, "softMinimumEnabled": False,
                            "softMaximumEnabled": False, **rule})


def saved_model_id(p):
    return next(m["id"] for m in p.list_saved_models() if m["name"] == "load_forecast")


def ensure_scenario(p):
    sm, pk = saved_model_id(p), PROJECT_KEY
    mes = p.list_model_evaluation_stores()[0].mes_id
    item = lambda kind, i: {"type": kind, "projectKey": pk, "itemId": i, "partitionsSpec": ""}  # noqa: E731
    steps = [
        {"id": "dq", "type": "check_dataset", "name": "Data quality rules on gold_features",
         "params": {"checks": [item("DATASET", "gold_features")], "proceedOnFailure": False}},
        {"id": "retrain", "type": "build_flowitem", "name": "Retrain Visual ML model load_forecast",
         "params": {"builds": [item("SAVED_MODEL", sm)], "jobType": "NON_RECURSIVE_FORCED_BUILD"}},
        {"id": "outputs", "type": "build_flowitem", "name": "Score, evaluate, backtest, tests, anomalies, charts",
         "params": {"builds": [item("DATASET", d) for d in MONITORED] + [item("MODEL_EVALUATION_STORE", mes)],
                    "jobType": "RECURSIVE_BUILD"}},
        {"id": "gate", "type": "custom_python", "name": "Quality gate: model must beat seasonal naive",
         "params": {"script": (ROOT / "dataiku/recipes/scenario_quality_gate.py").read_text(),
                    "envSelection": {"envMode": "EXPLICIT_ENV", "envName": CODE_ENV}}},
    ]
    for s in steps:
        s.setdefault("runConditionType", "RUN_IF_STATUS_MATCH")
        s.setdefault("runConditionStatuses", ["SUCCESS", "WARNING"])
    existing = [s for s in p.list_scenarios() if s["name"] == "Monthly retrain"]
    sc = p.get_scenario(existing[0]["id"]) if existing else p.create_scenario("Monthly retrain", "step_based")
    st = sc.get_settings()
    st.raw_steps.clear()
    st.raw_steps.extend(steps)
    st.raw_triggers.clear()
    st.add_monthly_trigger(day=1, hour=3, minute=0, timezone="Europe/Paris")
    st.active = True
    try:
        st.save()
    except dataikuapi.utils.DataikuException as e:
        if "LicenseRestriction" not in str(e):
            raise
        # Free Edition: no time-based triggers. Keep the scenario, run it manually / via API.
        st.raw_triggers.clear()
        st.active = False
        st.save()
        log("scenario saved without its monthly trigger (not available on this licence)")
    return sc


def ensure_wiki(p):
    wiki = p.get_wiki()
    titles = {a.get_data().get_name(): a for a in wiki.list_articles()}
    pages = {"Project handover": ROOT / "dataiku/WIKI.md",
             "Synthese (FR)": ROOT / "docs/synthese_fr.md",
             "How this project is built": ROOT / "dataiku/FLOW_GUIDE.md"}
    for title, path in pages.items():
        article = titles.get(title) or wiki.create_article(title)
        data = article.get_data()
        data.set_body(path.read_text())
        data.save()
    settings = wiki.get_settings()
    home = next(a for a in wiki.list_articles() if a.get_data().get_name() == "Project handover")
    settings.set_home_article_id(home.article_id)
    settings.save()


def stage_ops(p):
    ensure_dq_rules(p)
    ensure_scenario(p)
    ensure_wiki(p)
    log("data quality rules, scenario 'Monthly retrain', wiki")


def wait_job(job):
    while (state := job.get_status()["baseStatus"]["state"]) not in ("DONE", "FAILED", "ABORTED"):
        time.sleep(5)
    if state != "DONE":
        raise RuntimeError(f"job {job.id} {state}")


def stage_retrain(p):
    """The 'Monthly retrain' scenario's steps, run through the API (Free Edition cannot trigger
    scenarios programmatically): DQ rules -> retrain -> rebuild outputs -> quality gate."""
    dq = p.get_dataset("gold_features").get_data_quality_rules()
    started = time.time() * 1000
    dq.compute_rules()
    for _ in range(60):  # rules are computed asynchronously
        results = {r.get_raw().get("displayName"): r.get_last_result() for r in dq.list_rules()}
        if all(res is not None and res.data.get("computeDate", 0) >= started for res in results.values()):
            break
        time.sleep(5)
    failed = [name for name, res in results.items() if res is None or res.outcome not in ("OK", "WARNING")]
    if failed:
        raise RuntimeError(f"data quality rules failed: {failed}")
    log("data quality rules: all OK")
    item = lambda kind, i: {"type": kind, "id": i, "projectKey": PROJECT_KEY}  # noqa: E731
    wait_job(p.start_job({"type": "NON_RECURSIVE_FORCED_BUILD", "outputs": [item("SAVED_MODEL", saved_model_id(p))]}))
    log("retrained load_forecast")
    mes = p.list_model_evaluation_stores()[0].mes_id
    outputs = [item("DATASET", d) for d in MONITORED] + [item("MODEL_EVALUATION_STORE", mes)]
    wait_job(p.start_job({"type": "RECURSIVE_BUILD", "outputs": outputs}))
    log("rebuilt monitored outputs and model evaluation")
    ds = p.get_dataset("metrics_overall")
    cols = [c["name"] for c in ds.get_schema()["columns"]]
    mase = {r[cols.index("model")]: float(r[cols.index("median_mase")]) for r in ds.iter_rows()}
    msg = f"median MASE lgbm={mase['lgbm']:.3f} vs seasonal naive={mase['snaive_168']:.3f}"
    if mase["lgbm"] >= 0.95 * mase["snaive_168"]:
        raise RuntimeError("quality gate failed: " + msg)
    log("quality gate passed: " + msg)
    n_evals = len(p.list_model_evaluation_stores()[0].list_model_evaluations())
    log(f"model evaluation store now holds {n_evals} evaluations")


# ---------------------------------------------------------------- dashboard
TABLES = {"metrics_overall": "Accuracy by model (12 monthly refits, 2017)",
          "dm_tests": "Diebold-Mariano verdict per meter and comparison",
          "interval_scores": "P10-P90 coverage per site and month (target 0.80)",
          "consumption_anomalies": "Consumption anomalies, ranked by size",
          "meter_faults": "Meter faults for the metering team"}
INTRO = """### Energy load forecasting: day-ahead, 76 meters
Best model: **mean of LightGBM + GRU**, -35% error vs last week's profile, significantly better than
LightGBM alone on 63/76 meters. P10-P90 bands are conformally calibrated (79% coverage).
Weather barely matters on these gas-heated campuses (7/76 meters are temperature-sensitive)."""


def text_tile(text, box):
    return {"tileType": "TEXT", "box": box, "tileParams": {"text": text, "textAlign": "LEFT"},
            "titleOptions": {"showTitle": "NO"}}


def insight_tile(insight, box):
    return {"tileType": "INSIGHT", "insightId": insight["id"], "insightType": insight["type"],
            "displayMode": "INSIGHT", "box": box, "tileParams": {}, "resizeImageMode": "FIT_SIZE",
            "titleOptions": {"showTitle": "YES"}, "clickAction": "DO_NOTHING", "autoLoad": True}


def box(left, top, width, height):
    return {"left": left, "top": top, "width": width, "height": height}


def stage_dashboard(p):
    insights = {i["name"]: i for i in p.list_insights(as_type="listitems")}
    for ds, label in TABLES.items():
        if label not in insights:
            p.create_insight({"type": "dataset_table", "name": label, "params": {"datasetSmartName": ds}})
    insights = {i["name"]: i for i in p.list_insights(as_type="listitems")}
    fig = lambda label: insights[label]  # noqa: E731  labels set in publish_insights.py
    pages = [
        {"title": "Forecast performance", "grid": {"tiles": [
            text_tile(INTRO, box(0, 0, 36, 4)),
            insight_tile(fig("Forecast error by model (MASE per meter)"), box(0, 4, 20, 12)),
            insight_tile(insights[TABLES["metrics_overall"]], box(20, 4, 16, 12)),
            insight_tile(fig("Day-ahead forecast vs actual, one week"), box(0, 16, 20, 9)),
            insight_tile(insights[TABLES["dm_tests"]], box(20, 16, 16, 9)),
        ]}},
        {"title": "Uncertainty and weather", "grid": {"tiles": [
            insight_tile(fig("Calibrated P10-P90 forecast, one week"), box(0, 0, 22, 10)),
            insight_tile(insights[TABLES["interval_scores"]], box(22, 0, 14, 10)),
            insight_tile(fig("Energy signature: consumption vs temperature"), box(0, 10, 22, 10)),
            text_tile("Only buildings heated electrically respond to temperature (left: +20% load per "
                      "degree below 17 C). Most campus buildings are flat (right): weather adds little.",
                      box(22, 10, 14, 10)),
        ]}},
        {"title": "Monitoring", "grid": {"tiles": [
            insight_tile(fig("Largest building-specific anomaly"), box(0, 0, 36, 9)),
            insight_tile(insights[TABLES["consumption_anomalies"]], box(0, 9, 20, 12)),
            insight_tile(insights[TABLES["meter_faults"]], box(20, 9, 16, 12)),
        ]}},
    ]
    existing = [d for d in p.list_dashboards(as_type="listitems") if d.name == "Energy forecasting"]
    dash = existing[0].to_dashboard() if existing else p.create_dashboard("Energy forecasting")
    st = dash.get_settings()
    st.get_raw()["pages"] = pages
    st.get_raw()["listed"] = True
    st.save()
    log(f"dashboard 'Energy forecasting': {len(pages)} pages, {sum(len(pg['grid']['tiles']) for pg in pages)} tiles")


# ---------------------------------------------------------------- export
def stage_export(p):
    """Project definition only (Flow, recipes, ML task, scenario, dashboard, wiki), no data or model
    binaries: those are rebuilt by running the Flow. Importable in any DSS 15+."""
    out = ROOT / "dataiku/export" / f"{PROJECT_KEY}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    options = {"exportUploads": False, "exportManagedFS": False, "exportAnalysisModels": False,
               "exportSavedModels": False, "exportModelEvaluationStores": False, "exportManagedFolders": False,
               "exportAllInputDatasets": False, "exportAllDatasets": False, "exportGitRepository": False,
               "exportInsightsData": True}
    with p.get_export_stream(options) as stream, open(out, "wb") as fh:
        for chunk in stream.stream(512 * 1024):
            fh.write(chunk)
    log(f"exported {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


STAGES = {"setup": stage_setup, "flow": stage_flow, "ml": stage_ml, "run": stage_run, "ops": stage_ops,
          "dashboard": stage_dashboard, "export": stage_export, "retrain": stage_retrain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    c = client()
    for stage in args.stages.split(","):
        log(f"== {stage}")
        if stage == "setup":
            stage_setup(c, args.recreate)
        else:
            STAGES[stage](c.get_project(PROJECT_KEY))


if __name__ == "__main__":
    sys.exit(main())
