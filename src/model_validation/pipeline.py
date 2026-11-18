"""Diagnose the saved 90% logistic model; never retrain or recalibrate it.

Run normally in PyCharm. Uses the saved 10% holdouts (2012-2017) and the
already-inspected 2020 evaluation. 2021-2023 loan observations stay reserved.
Writes results/model_validation/report.md and calibration PNGs for 2017/2020.
Source event verification reads five performance columns for positive loans
in those two focus groups; this can be the slow part and is cached per file.
Optional --skip-event-history produces core diagnostics with that audit marked
SKIPPED. Rerun normally to complete the audit using saved progress.

No source data/model weights change. Small prediction caches live under
data/model_validation/. Diagnostic calibration regressions assess errors;
their coefficients are NOT used to change the model or its probabilities.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from logistic_regression.cohort_audit import POLICY, select_baseline
from logistic_regression.evaluation import evaluate
from logistic_regression.preprocessing import NUMERIC, CATEGORICAL, atomic_json, identifier as q, literal, projection
from logistic_regression.train import digest, unpack
from logistic_regression.walk_forward import create_source, source_files
from logistic_regression.year_stratified import pool_filter
if __package__:
    from .diagnostics import (SEGMENTS, auc_interval, calibration_diagnostics, date_audit, drift,
                              month_resampling, plot_calibration, summary_groups)
    from .events import audit_events, fingerprint, rows
    from .report import write_report
else:
    from diagnostics import (SEGMENTS, auc_interval, calibration_diagnostics, date_audit, drift,
                             month_resampling, plot_calibration, summary_groups)
    from events import audit_events, fingerprint, rows
    from report import write_report


VERSION = 1
GROUPS = tuple(f"holdout_{year}" for year in range(2012, 2018)) + ("oot_2020",)
FOCUS = ("holdout_2017", "oot_2020")
AUDIT = ["origination_cohort", "horizon_end", "event_in_window", "event_report_month",
         "first_exit_month", "future_months", "future_unknown_months", "internal_gap_months",
         "reclassified_as_repaid", "payoff_confirmation_month"]
CACHE_COLUMNS = ["loan_id", "observation_month", "candidate_default_12m", "predicted_pd", "group_name",
                 "assumed_label_available_month"] + NUMERIC + CATEGORICAL + AUDIT


def select_model(prepared_run, experiment):
    inputs, baseline, baseline_path = select_baseline(prepared_run)
    checkpoint = baseline_path.parent.parent / experiment / "metrics.json"
    model = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (not model.get("status", "").startswith("COMPLETED") or not model.get("training", {}).get("converged")
            or model["signature"].get("holdout_fraction") != 0.10
            or model.get("prepared_run") != inputs["prepared_run"]
            or model["signature"].get("years") != list(range(2012, 2018))
            or model["signature"].get("oot_year") != 2020
            or model["signature"].get("model_date") != "2020-01-01"
            or model["signature"]["input_manifest_sha256"] != baseline["signature"]["input_manifest_sha256"]):
        raise ValueError("Select a complete, converged 2012-2017 90/10 model with 2020 evaluation")
    if set(model.get("evaluations", {})) != set(GROUPS):
        raise ValueError("The model checkpoint does not contain exactly the allowed evaluation groups")
    files, sources = source_files(inputs, last_year=2020)
    if sources != model["signature"]["files"]:
        raise ValueError("Corrected candidate data changed since the model was evaluated")
    for path in files:
        if not set(AUDIT).issubset(pq.read_schema(path).names):
            raise ValueError(f"Missing audit fields in {path}")
    weights = ROOT / "data" / "models" / "logistic_regression" / inputs["prepared_run"] / POLICY / experiment / "model_weights.npz"
    if digest(weights) != model["model_sha256"]:
        raise ValueError("Saved model weights changed")
    expressions, features = projection(model["fitted"])
    with np.load(weights, allow_pickle=False) as saved:
        coefficients, intercept = saved["coefficients"].copy(), float(saved["intercept"].item())
        if (saved["feature_names"].tolist() != features or features != model["feature_names"]
                or coefficients.shape != (len(features),) or not np.isfinite(coefficients).all()
                or not np.isfinite(intercept)):
            raise ValueError("Model coefficients and preprocessing whitelist differ")
    split_folder = ROOT / "data" / "model_inputs" / "logistic_regression" / inputs["prepared_run"] / POLICY / experiment
    split_paths, splits = [], []
    for year in range(2012, 2018):
        path = split_folder / f"split_{year}.parquet"
        if digest(path) != model["splits"][str(year)]["sha256"]:
            raise ValueError(f"Saved {year} split assignments changed")
        split_paths.append(path)
        splits.append(fingerprint(path))
    preparation_path = ROOT / "results" / "data_preparation" / inputs["prepared_run"] / "manifest.json"
    prepared = json.loads(preparation_path.read_text(encoding="utf-8"))
    provenance = {"model_checkpoint": fingerprint(checkpoint), "checkpoint_sha256": digest(checkpoint),
                  "weights": fingerprint(weights), "model_sha256": model["model_sha256"],
                  "sources": sources, "splits": splits, "preparation_manifest_sha256": digest(preparation_path)}
    return inputs, model, prepared, files, split_paths, expressions, features, coefficients, intercept, provenance


def setup_views(connection, files, split_paths):
    create_source(connection, files, cutoff="2021-01-01")
    paths = "[" + ", ".join(literal(path.as_posix()) for path in split_paths) + "]"
    connection.execute(f"CREATE TEMP VIEW assignments AS SELECT * FROM read_parquet({paths})").fetchall()
    connection.execute("CREATE TEMP VIEW training AS SELECT s.* FROM source s JOIN assignments a USING (loan_id) "
                       "WHERE NOT a.is_holdout AND " + pool_filter()).fetchall()
    candidates = "[" + ", ".join(literal(path.as_posix()) for path in files) + "]"
    connection.execute("CREATE TEMP VIEW candidate_audit AS SELECT loan_id, " + ", ".join(map(q, AUDIT))
                       + f" FROM read_parquet({candidates}) WHERE observation_month < DATE '2021-01-01'").fetchall()


def cache_predictions(connection, key, target, cached, expected, expressions, features, coefficients, intercept):
    if cached is not None:
        if (not target.is_file() or fingerprint(target) != cached["output"]
                or pq.read_metadata(target).num_rows != expected["loans"]
                or pq.read_schema(target).names != CACHE_COLUMNS):
            raise ValueError(f"Prediction cache changed or is missing: {target}")
        print(f"Reusing prediction cache: {key}", flush=True)
        return cached
    if target.exists():
        raise ValueError(f"Uncheckpointed prediction cache: {target}. Select a new --name.")
    year = int(key.rsplit("_", 1)[1])
    if key.startswith("holdout_"):
        source = ("SELECT s.* FROM source s JOIN assignments a USING (loan_id) WHERE a.is_holdout AND "
                  + pool_filter() + f" AND year(s.observation_month)={year}")
    else:
        if year != 2020:
            raise ValueError("Reserved data cannot be evaluated")
        source = ("SELECT * FROM source WHERE observation_month>=DATE '2020-01-01' "
                  "AND observation_month<DATE '2021-01-01' AND candidate_default_12m IN (0,1) "
                  "AND assumed_label_available_month IS NOT NULL")
    connection.execute("CREATE OR REPLACE TEMP VIEW selected AS SELECT *, 'validation' AS proposed_split FROM ("
                       + source + ")").fetchall()
    extra = [q(name) for name in NUMERIC+CATEGORICAL+AUDIT+["assumed_label_available_month"]] + [f"{literal(key)} AS group_name"]
    query = "SELECT " + ", ".join(expressions+extra) + " FROM selected s JOIN candidate_audit m USING (loan_id)"
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
    writer = None
    count = events = 0
    probability_sum = 0.0
    print(f"Scoring frozen model for {key}: {expected['loans']:,} loans; no training ...", flush=True)
    try:
        with connection.sql(query).to_arrow_reader(batch_size=65_536) as reader:
            for batch in reader:
                matrix, labels, dates = unpack(batch, features)
                if (np.any(dates < np.datetime64(f"{year}-01-01"))
                        or np.any(dates >= np.datetime64(f"{year+1}-01-01"))):
                    raise ValueError("Wrong observation year reached the evaluation cache")
                probabilities = expit(np.asarray(matrix @ coefficients).ravel()+intercept)
                if not np.isfinite(probabilities).all():
                    raise ValueError("Non-finite model probabilities")
                data = {name: batch.column(batch.schema.get_field_index(name)) for name in CACHE_COLUMNS if name != "predicted_pd"}
                data["predicted_pd"] = pa.array(probabilities)
                table = pa.table({name: data[name] for name in CACHE_COLUMNS})
                if writer is None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                count += len(labels)
                events += int(labels.sum())
                probability_sum += float(probabilities.sum())
        if writer is not None:
            writer.close()
            writer = None
        if (count != expected["loans"] or events != expected["bad_events"]
                or not np.isclose(probability_sum/count, expected["mean_predicted_pd"], rtol=1e-9, atol=1e-12)
                or pq.read_metadata(temporary).num_rows != count):
            raise ValueError("Cached predictions do not reproduce the saved model evaluation")
        temporary.rename(target)
        return {"output": fingerprint(target), "rows": count, "events": events,
                "mean_pd": probability_sum/count}
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def diagnose(connection, key, path, training_profile, training_rate):
    connection.execute(f"CREATE OR REPLACE TEMP VIEW evaluation AS SELECT * FROM read_parquet({literal(path.as_posix())})").fetchall()
    grouped = summary_groups(connection, "evaluation")
    checks = date_audit(connection)
    event_calendar = rows(connection, """
        SELECT strftime(event_in_window,'%Y-%m') AS event_month, count(*) AS loans
        FROM evaluation WHERE candidate_default_12m=1 GROUP BY event_in_window ORDER BY event_in_window
    """)
    table = pq.read_table(path, columns=["candidate_default_12m", "predicted_pd"])
    y = table["candidate_default_12m"].to_numpy(zero_copy_only=False).astype(np.int8)
    p = table["predicted_pd"].to_numpy(zero_copy_only=False)
    metrics, bands = evaluate(y, p, training_rate)
    print(f"Calibration and uncertainty diagnostics: {key}", flush=True)
    return {"metrics": metrics, "bands": bands, "calibration": calibration_diagnostics(y, p),
            "auc_interval": auc_interval(y, p), "month_resampling": month_resampling(grouped),
            "groups": grouped, "drift": drift(training_profile, grouped),
            "date_checks": checks, "event_calendar": event_calendar}


def main(prepared_run=None, experiment="year_stratified_90_10", name="diagnostics_90_10",
         threads=2, memory_limit="2GB", skip_event_history=False):
    if (not re.fullmatch(r"year_stratified(?:_[a-z0-9_]{1,32})?", experiment)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,47}", name)):
        raise ValueError("Invalid experiment/diagnostic name")
    inputs, model, prepared, files, splits, expressions, features, coefficients, intercept, provenance = select_model(prepared_run, experiment)
    folder = ROOT / "results" / "model_validation" / inputs["prepared_run"] / name
    data_folder = ROOT / "data" / "model_validation" / inputs["prepared_run"] / name
    checkpoint = folder / "manifest.json"
    report_path = ROOT / "results" / "model_validation" / "report.md"
    signature = {"version": VERSION, "experiment": experiment, "provenance": provenance,
                 "groups": list(GROUPS), "focus": list(FOCUS), "numpy": np.__version__,
                 "pyarrow": pa.__version__, "duckdb": duckdb.__version__,
                 "scipy": scipy.__version__, "sklearn": sklearn.__version__, "segments": SEGMENTS,
                 "oe_resampling": {"unit": "starting_month", "repetitions": 2000, "seed": 42}}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Model/data/settings changed. Use --name diagnostics_90_10_v2 to preserve saved work.")
    else:
        state = {"prepared_run": inputs["prepared_run"], "signature": signature,
                 "predictions": {}, "diagnostics": {}, "event_source": {"files": {}}, "plots": {}}
    state.update(status="RUNNING - DIAGNOSTICS ONLY", model_experiment=experiment,
                 training_rows=model["training"]["rows"], training_rate=model["training"]["event_rate"])
    state.pop("error", None)

    def verify_sources():
        for record in provenance["sources"]+provenance["splits"]+[provenance["weights"], provenance["model_checkpoint"]]:
            if fingerprint(Path(record["path"])) != {key: value for key, value in record.items() if key != "rows"}:
                raise ValueError("An input/model/checkpoint changed during diagnostics")

    def save():
        atomic_json(checkpoint, state)
        write_report(report_path, state)

    save()
    try:
        pa.set_cpu_count(threads)
        with duckdb.connect(config={"memory_limit": memory_limit, "threads": str(threads)}) as connection, threadpool_limits(limits=threads):
            connection.execute("SET enable_progress_bar=false").fetchall()
            setup_views(connection, files, splits)
            if "training_profile" not in state:
                print("Counting training-population feature distributions; no fitting ...", flush=True)
                state["training_profile"] = summary_groups(connection, "training", outcomes=False)
                if any(sum(row["loans"] for row in state["training_profile"] if row["dimension"]==dimension)
                       != model["training"]["rows"] for dimension in {row["dimension"] for row in state["training_profile"]}):
                    raise ValueError("Training profile differs from saved model membership")
                verify_sources()
                save()
            prediction_paths = {}
            for key in GROUPS:
                target = data_folder / f"predictions_{key}.parquet"
                state["predictions"][key] = cache_predictions(connection, key, target,
                    state["predictions"].get(key), model["evaluations"][key]["metrics"],
                    expressions, features, coefficients, intercept)
                prediction_paths[key] = target
                verify_sources()
                save()
                if key not in state["diagnostics"]:
                    state["diagnostics"][key] = diagnose(connection, key, target, state["training_profile"], state["training_rate"])
                    verify_sources()
                    save()
                else:
                    print(f"Reusing diagnostics: {key}", flush=True)
                if key in FOCUS:
                    plot = folder / f"calibration_{key}.png"
                    if not plot.is_file():
                        plot_calibration(plot, state["diagnostics"][key]["bands"], key.replace("_", " "))
                    state["plots"][key] = plot.relative_to(report_path.parent).as_posix()
                    save()
            if skip_event_history:
                state["event_source"]["status"] = "SKIPPED - rerun without --skip-event-history to complete"
                save()
            else:
                audit_events(connection, [prediction_paths[key] for key in FOCUS], prepared, ROOT,
                             data_folder / "events", state["event_source"], save)
            verify_sources()
        state["status"] = ("COMPLETED CORE DIAGNOSTICS - SOURCE EVENT AUDIT SKIPPED" if skip_event_history
                           else "COMPLETED - REVIEW FINDINGS; MODEL NOT UPDATED")
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    except Exception as error:
        state.update(status="FAILED - SAVED PROGRESS RETAINED", error=str(error))
        save()
        raise
    print(f"Finished. OPEN THIS REPORT: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run")
    parser.add_argument("--experiment", default="year_stratified_90_10")
    parser.add_argument("--name", default="diagnostics_90_10")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--skip-event-history", action="store_true")
    args = parser.parse_args()
    main(args.prepared_run, args.experiment, args.name, args.threads, args.memory_limit, args.skip_event_history)
