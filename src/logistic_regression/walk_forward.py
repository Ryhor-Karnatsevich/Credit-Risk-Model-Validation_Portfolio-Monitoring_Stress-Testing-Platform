"""Evaluate the unchanged logistic specification across historical years.

Run normally in PyCharm. Evaluates starting observations in 2016, 2017, 2018;
each model uses earlier loans with outcomes available before that year starts.
Preprocessing is refitted inside each training period. Evaluation outcomes
must be available before 2020, so the 2018 evaluation ends in November.
No 2020-2023 observations are fitted, scored or evaluated.

Reads corrected candidate Parquets, not monthly histories or globally encoded
inputs. No cleaning or conversion. Completed compatible folds are reused.
Main report: results/logistic_regression/walk_forward.md
Optional: --prepared-run ID --name walk_forward_v2 --threads 1
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from uuid import uuid4
import warnings

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.sparse import csr_matrix
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

if __package__:
    from .cohort_audit import POLICY, ROOT, local_path, select_baseline
    from .evaluation import evaluate, number
    from .preprocessing import NUMERIC, CATEGORICAL, atomic_json, fit_preprocessing, identifier, literal, projection
    from .train import digest, save_weights, unpack
else:
    from cohort_audit import POLICY, ROOT, local_path, select_baseline
    from evaluation import evaluate, number
    from preprocessing import NUMERIC, CATEGORICAL, atomic_json, fit_preprocessing, identifier, literal, projection
    from train import digest, save_weights, unpack


YEARS = (2016, 2017, 2018)
START = "2012-01-01"
REVIEW_CUTOFF = "2020-01-01"
VERSION = 1
BATCH_SIZE = 65_536


def fold_filters(year):
    """Use calendar dates and label availability, never saved global split flags."""
    if year not in YEARS:
        raise ValueError("Only the predefined pre-test evaluation years are allowed")
    known = "candidate_default_12m IN (0, 1)"
    training = (f"{known} AND observation_month >= DATE '{START}' "
                f"AND observation_month < DATE '{year}-01-01' "
                f"AND assumed_label_available_month < DATE '{year}-01-01'")
    evaluation = (f"{known} AND observation_month >= DATE '{year}-01-01' "
                  f"AND observation_month < DATE '{year + 1}-01-01' "
                  f"AND assumed_label_available_month < DATE '{REVIEW_CUTOFF}'")
    return training, evaluation


def create_fold_views(connection, year):
    training, evaluation = fold_filters(year)
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW eligible AS
        SELECT *, CASE WHEN {training} THEN 'development' ELSE 'validation' END AS proposed_split
        FROM source WHERE ({training}) OR ({evaluation})
    """)
    rows = connection.execute("""
        SELECT proposed_split, count(*), sum(candidate_default_12m),
            min(observation_month), max(observation_month), max(assumed_label_available_month)
        FROM eligible GROUP BY proposed_split
    """).fetchall()
    counts = {split: {"rows": count, "events": int(events), "event_rate": events / count,
                       "first_observation": str(first), "last_observation": str(last),
                       "last_label_available": str(available)}
              for split, count, events, first, last, available in rows}
    if set(counts) != {"development", "validation"}:
        raise ValueError(f"Year {year}: training and evaluation must both contain loans")
    if not 0 < counts["development"]["events"] < counts["development"]["rows"]:
        raise ValueError(f"Year {year}: training needs both outcome classes")
    if (counts["development"]["last_label_available"] >= f"{year}-01-01"
            or counts["validation"]["last_label_available"] >= REVIEW_CUTOFF):
        raise ValueError("Future outcome availability reached the wrong period")
    return counts


def source_files(inputs, last_year=None):
    required = set(NUMERIC + CATEGORICAL + ["loan_id", "observation_month",
                   "candidate_default_12m", "assumed_label_available_month", "target_policy"])
    files, fingerprints = [], []
    # Defaults preserve the original walk-forward scope. Other experiments
    # explicitly select their own final source year.
    last_year = max(YEARS) if last_year is None else last_year
    for year in range(2012, last_year + 1):
        update = inputs["label_updates"].get(str(year))
        if update is None:
            raise ValueError(f"Missing corrected candidate table for {year}")
        if any(update["stats"].get(key) != 0 for key in
               ("lost_default_labels", "changed_known_labels", "unsupported_zero_labels")):
            raise ValueError(f"Repayment-policy checks failed for {year}")
        path = local_path(update["file"])
        expected = ROOT / "data" / "prepared" / inputs["prepared_run"] / "target_revisions" / POLICY / str(year) / "cohort_candidates.parquet"
        if path != expected.resolve():
            raise ValueError(f"Unexpected corrected source path: {path}")
        metadata, info = pq.read_metadata(path), path.stat()
        if (not required.issubset(metadata.schema.names) or metadata.num_rows != update["stats"]["rows"]
                or info.st_size != update["bytes"]):
            raise ValueError(f"Candidate file differs from saved preprocessing: {path}")
        files.append(path)
        fingerprints.append({"path": str(path), "bytes": info.st_size,
                             "mtime_ns": info.st_mtime_ns, "rows": metadata.num_rows})
    return files, fingerprints


def create_source(connection, files, cutoff=REVIEW_CUTOFF):
    paths = "[" + ", ".join(literal(path.as_posix()) for path in files) + "]"
    numeric = [f"CASE WHEN isfinite({identifier(name)}) THEN CAST({identifier(name)} AS DOUBLE) END AS {identifier(name)}"
               for name in NUMERIC]
    categorical = []
    for name in CATEGORICAL:
        value = ("CASE WHEN borrowers_code = 1 THEN 'single' WHEN borrowers_code >= 2 THEN 'multiple' END"
                 if name == "borrowers_code" else f"nullif(trim(CAST({identifier(name)} AS VARCHAR)), '')")
        categorical.append(f"{value} AS {identifier(name)}")
    metadata = ["loan_id", "observation_month", "candidate_default_12m", "assumed_label_available_month", "target_policy"]
    connection.execute("CREATE TEMP VIEW source AS SELECT " + ", ".join(metadata + numeric + categorical)
                       + f" FROM read_parquet({paths}) WHERE observation_month >= DATE '{START}' "
                       + f"AND observation_month < DATE {literal(cutoff)}")
    # Exhaust the scalar result before later relation/Arrow reads use this connection.
    invalid = connection.execute(f"""
        SELECT count(*) FROM source WHERE target_policy IS DISTINCT FROM {literal(POLICY)}
            OR (candidate_default_12m IS NOT NULL AND candidate_default_12m NOT IN (0, 1))
    """).fetchall()[0][0]
    if invalid:
        raise ValueError("Invalid labels/policy in corrected pre-test candidate data")


def encoded_batches(connection, expressions, split):
    query = "SELECT " + ", ".join(expressions) + " FROM eligible WHERE proposed_split = " + literal(split)
    with connection.sql(query).to_arrow_reader(batch_size=BATCH_SIZE) as reader:
        yield from reader


def training_matrix(connection, expressions, features, count, year, memory_gb):
    max_nonzero = 2 * len(NUMERIC) + len(CATEGORICAL)
    capacity = count * max_nonzero
    storage = capacity * 12 + (count + 1) * 4 + count
    if capacity >= np.iinfo(np.int32).max or 2 * storage + 512 * 1024**2 > memory_gb * 1024**3:
        raise MemoryError("Sparse training allocation guard exceeded; review available RAM")
    values = np.empty(capacity, dtype=np.float64)
    indices = np.empty(capacity, dtype=np.int32)
    pointers = np.empty(count + 1, dtype=np.int32)
    targets = np.empty(count, dtype=np.int8)
    pointers[0] = 0
    offset = nonzero = 0
    for batch in encoded_batches(connection, expressions, "development"):
        matrix, labels, dates = unpack(batch, features)
        end = offset + len(labels)
        if (end > count or nonzero + matrix.nnz > capacity
                or np.max(np.diff(matrix.indptr)) > max_nonzero
                or np.any(dates < np.datetime64(START)) or np.any(dates >= np.datetime64(f"{year}-01-01"))):
            raise ValueError("Training counts, dates or sparsity differ from the fold design")
        values[nonzero:nonzero + matrix.nnz] = matrix.data
        indices[nonzero:nonzero + matrix.nnz] = matrix.indices
        pointers[offset + 1:end + 1] = matrix.indptr[1:] + nonzero
        targets[offset:end] = labels
        offset, nonzero = end, nonzero + matrix.nnz
    if offset != count or np.unique(targets).size != 2:
        raise ValueError("Training row count/classes differ from the fold summary")
    return csr_matrix((values[:nonzero], indices[:nonzero], pointers),
                      shape=(count, len(features)), copy=False), targets


def score_fold(connection, expressions, features, coefficients, intercept, count, year, training_rate):
    labels = np.empty(count, dtype=np.int8)
    probabilities = np.empty(count, dtype=np.float64)
    offset = 0
    for batch in encoded_batches(connection, expressions, "validation"):
        matrix, y, dates = unpack(batch, features)
        end = offset + len(y)
        if (end > count or np.any(dates < np.datetime64(f"{year}-01-01"))
                or np.any(dates >= np.datetime64(f"{year + 1}-01-01"))):
            raise ValueError("Evaluation rows/dates differ from the fold design")
        labels[offset:end] = y
        probabilities[offset:end] = expit(np.asarray(matrix @ coefficients).ravel() + intercept)
        offset = end
    if offset != count:
        raise ValueError("Evaluation row count differs from the fold summary")
    metrics, bands = evaluate(labels, probabilities, training_rate)
    return {"metrics": metrics, "bands": bands}


def write_report(path, state):
    lines = ["# Logistic regression across years", "", f"Status: {state['status']}",
             f"Preparation run: {state['prepared_run']}", "",
             "Each row is a separately fitted model with the same logistic specification. Training expands over time; the models are not tuned to these results.", "",
             "## Design", "",
             "- Evaluate starting observations in 2016, 2017 and 2018. Years refer to first observation, not loan origination.",
             "- Fit only earlier observations whose 12-month outcomes plus the saved reporting lag are available before the evaluation year starts.",
             "- Refit medians, scaling and category lists using that fold's training loans only. Missing/unseen categories map to unknown.",
             "- Use the existing verified-repayment labels. Original global eligibility flags are replaced by fold-specific date rules, so previously withheld 2017 observations can be evaluated.",
             "- Evaluation outcomes must be available before 2020. This limits the 2018 evaluation to January-November under the current reporting-lag assumption.",
             "- Final-test observations from 2020-2023 are not fitted, scored or evaluated. Their origination files are not opened.",
             "- No sampling, class balancing, probability recalibration or hyperparameter search. These are historical robustness checks, not untouched final-test results.", "",
             "## Results", "",
             "| Evaluation year | Training loans | Evaluation loans | Bad events | ROC AUC | Mean predicted risk | Observed risk | Observed / expected | Top 10% event capture | Converged |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for year, fold in sorted(state["folds"].items()):
        if fold.get("status") != "COMPLETE":
            continue
        m, training = fold["evaluation"]["metrics"], fold["training"]
        lines.append(f"| {year} | {training['rows']:,} | {m['loans']:,} | {m['bad_events']:,} | "
                     f"{number(m['roc_auc'])} | {number(m['mean_predicted_pd'], True)} | "
                     f"{number(m['observed_event_rate'], True)} | {number(m['observed_expected_ratio'])} | "
                     f"{number(m['top_10pct_event_capture'], True)} | {'Yes' if training['converged'] else 'NO'} |")
    lines += ["", "## Dates and probability errors", "",
              "Lower log loss/Brier is better. Each constant benchmark predicts that fold's training event rate for every evaluation loan.", "",
              "| Year | Training observations | Latest training outcome availability | Evaluation observations | Log loss / constant | Brier / constant |",
              "|---|---|---|---|---:|---:|"]
    for year, fold in sorted(state["folds"].items()):
        if fold.get("status") != "COMPLETE":
            continue
        t, v, m = fold["training"], fold["counts"]["validation"], fold["evaluation"]["metrics"]
        lines.append(f"| {year} | {t['first_observation']} to {t['last_observation']} | "
                     f"{t['last_label_available']} | {v['first_observation']} to {v['last_observation']} | "
                     f"{number(m['log_loss'])} / {number(m['constant_benchmark_log_loss'])} | "
                     f"{number(m['brier_score'])} / {number(m['constant_benchmark_brier'])} |")
    lines += ["", "## How to read this", "",
              "- ROC AUC measures ranking; 0.5 is random. It does not measure whether the probabilities are accurate.",
              "- Observed / expected above 1 means underprediction; below 1 means overprediction.",
              "- Compare years separately. Their populations, event rates and evaluation coverage differ; no pooled score or simple average proves stability.",
              "- Training periods overlap, so these results are not independent experiments. The 2018 period overlaps the baseline validation already inspected.",
              "- Nonconverged folds must be reviewed before drawing conclusions. No model-approval threshold is imposed.",
              "- Outcome availability uses the existing assumed reporting lag, not reconstructed historical release vintages.", ""]
    if state.get("error"):
        lines += ["## Error", "", state["error"].replace("\n", " "), ""]
    for year, fold in sorted(state["folds"].items()):
        if fold.get("status") != "COMPLETE":
            continue
        lines += [f"## {year}: risk groups", "",
                  "Band 1 contains the highest predicted risks; groups are approximately equal-sized.", "",
                  "| Band | Loans | Bad events | Predicted risk | Observed risk |", "|---|---:|---:|---:|---:|"]
        for band in fold["evaluation"]["bands"]:
            lines.append(f"| {band['risk_band']} | {band['loans']:,} | {band['bad_events']:,} | "
                         f"{number(band['mean_pd'], True)} | {number(band['observed_rate'], True)} |")
        lines.append("")
    lines += ["Fold settings, fitted preprocessing, timing and metrics are retained in the run checkpoint. Completed compatible folds are reused on rerun.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(prepared_run=None, name="walk_forward", threads=2, memory_gb=6.0):
    if not re.fullmatch(r"walk_forward(?:_[a-z0-9_]{1,32})?", name):
        raise ValueError("Use walk_forward or a name such as walk_forward_v2")
    if not np.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("The sparse allocation budget must be positive")
    inputs, baseline, baseline_path = select_baseline(prepared_run)
    files, fingerprints = source_files(inputs)
    config = {key: baseline["signature"][key] for key in ("C", "solver", "max_iter", "tolerance", "class_weight")}
    if config["solver"] != "lbfgs" or config["class_weight"] is not None:
        raise ValueError("This validation expects the existing unweighted L2/L-BFGS baseline")
    run_id = inputs["prepared_run"]
    folder = ROOT / "results" / "logistic_regression" / run_id / POLICY / name
    model_folder = ROOT / "data" / "models" / "logistic_regression" / run_id / POLICY / name
    checkpoint = folder / "metrics.json"
    report_path = ROOT / "results" / "logistic_regression" / "walk_forward.md"
    signature = {"version": VERSION, "years": list(YEARS), "start": START, "review_cutoff": REVIEW_CUTOFF,
                 "target_policy": POLICY, "input_manifest_sha256": baseline["signature"]["input_manifest_sha256"],
                 "files": fingerprints, "config": config, "threads": threads,
                 "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                 "duckdb": duckdb.__version__, "pyarrow": pa.__version__}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Inputs/settings changed. Use --name walk_forward_v2; earlier results are preserved.")
    else:
        state = {"prepared_run": run_id, "signature": signature, "folds": {},
                 "baseline_source": baseline_path.relative_to(ROOT).as_posix()}

    def save():
        atomic_json(checkpoint, state)
        write_report(report_path, state)

    def verify_sources():
        for item in fingerprints:
            info = Path(item["path"]).stat()
            if (info.st_size, info.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise ValueError("Corrected source changed during validation")
        if digest(baseline_path.parent.parent / "manifest.json") != signature["input_manifest_sha256"]:
            raise ValueError("Preprocessing manifest changed during validation")

    state.pop("error", None)
    state["status"] = "RUNNING - HISTORICAL VALIDATION ONLY"
    save()
    try:
        pa.set_cpu_count(threads)
        with duckdb.connect(config={"memory_limit": "2GB", "threads": str(threads)}) as connection, threadpool_limits(limits=threads):
            connection.execute("SET enable_progress_bar = false")
            create_source(connection, files)
            for year in YEARS:
                key = str(year)
                weights = model_folder / key / "model_weights.npz"
                fold = state["folds"].get(key, {})
                if weights.exists() and (not fold.get("model_sha256") or digest(weights) != fold["model_sha256"]):
                    raise ValueError(f"Year {year}: uncheckpointed/changed weights. Use a new experiment name.")
                if fold.get("status") == "COMPLETE":
                    if not weights.exists():
                        raise ValueError(f"Year {year}: completed fold weights are missing")
                    print(f"Reusing completed {year} evaluation.", flush=True)
                    continue
                print(f"\nEvaluation year {year}: preparing earlier training loans ...", flush=True)
                counts = create_fold_views(connection, year)
                if weights.exists():
                    fitted, features = fold["fitted"], fold["feature_names"]
                    expressions, names = projection(fitted)
                    with np.load(weights, allow_pickle=False) as saved:
                        coefficients = saved["coefficients"].copy()
                        intercept = float(saved["intercept"].item())
                        if saved["feature_names"].tolist() != features or names != features or fold["counts"] != counts:
                            raise ValueError("Saved fold preprocessing/counts differ from its model")
                    print("Reusing fitted weights; finishing evaluation only.", flush=True)
                else:
                    print("Learning medians, scales and categories from this training period only ...", flush=True)
                    fitted = fit_preprocessing(connection)
                    expressions, features = projection(fitted)
                    fold = {"status": "PREPARING", "counts": counts, "fitted": fitted, "feature_names": features}
                    state["folds"][key] = fold
                    save()
                    print(f"Loading {counts['development']['rows']:,} training loans ...", flush=True)
                    matrix, labels = training_matrix(connection, expressions, features,
                        counts["development"]["rows"], year, memory_gb)
                    estimator = LogisticRegression(C=config["C"], solver="lbfgs", l1_ratio=0.0,
                        max_iter=config["max_iter"], tol=config["tolerance"], class_weight=None)
                    print("Fitting logistic regression ...", flush=True)
                    started = time.monotonic()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        estimator.fit(matrix, labels)
                    training = dict(counts["development"])
                    training.update(fit_seconds=time.monotonic() - started,
                                    iterations=int(estimator.n_iter_.max()), features=len(features))
                    training["converged"] = (training["iterations"] < config["max_iter"]
                        and not any(issubclass(item.category, ConvergenceWarning) for item in caught))
                    if estimator.classes_.tolist() != [0, 1]:
                        raise ValueError("Unexpected fitted outcome classes")
                    coefficients, intercept = estimator.coef_[0].copy(), float(estimator.intercept_[0])
                    del matrix, labels, estimator
                    if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
                        raise ValueError("Non-finite fitted coefficients")
                    verify_sources()
                    weights.parent.mkdir(parents=True, exist_ok=True)
                    save_weights(weights, coefficients, intercept, features)
                    fold.update(status="FITTED", training=training, model_sha256=digest(weights),
                                warnings=[str(item.message) for item in caught])
                    save()
                if coefficients.shape != (len(features),) or not np.isfinite(coefficients).all() or not np.isfinite(intercept):
                    raise ValueError("Invalid saved coefficients")
                print(f"Scoring {counts['validation']['rows']:,} loans observed in {year} ...", flush=True)
                fold["evaluation"] = score_fold(connection, expressions, features, coefficients, intercept,
                    counts["validation"]["rows"], year, fold["training"]["event_rate"])
                verify_sources()
                fold.update(status="COMPLETE", completed_at_utc=datetime.now(timezone.utc).isoformat())
                save()
                print(f"Completed {year}; converged={fold['training']['converged']}.", flush=True)
        verify_sources()
        state["status"] = ("COMPLETED - HISTORICAL VALIDATION; REVIEW RESULTS"
            if all(fold["training"]["converged"] for fold in state["folds"].values())
            else "COMPLETED WITH NONCONVERGENCE - REVIEW REQUIRED")
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    except Exception as error:
        state.update(status="FAILED - COMPLETED FOLDS RETAINED", error=str(error))
        save()
        raise
    print(f"Finished. OPEN THIS REPORT: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run")
    parser.add_argument("--name", default="walk_forward")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--memory-gb", type=float, default=6.0)
    args = parser.parse_args()
    main(args.prepared_run, args.name, args.threads, args.memory_gb)
