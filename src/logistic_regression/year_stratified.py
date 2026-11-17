"""Fit one logistic model on 90% of each 2012-2017 observation year.

Run normally in PyCharm. The other 10% in each year is held out; the same
frozen model also evaluates loans first observed in 2020. Each yearly split
preserves the observed default proportion. Preprocessing learns only from
the combined 90% training groups. Training labels must be available before
2020-01-01. Earlier-year holdouts are retrospective, not historical forecasts.

This experiment explicitly uses 2020 for evaluation. 2021-2023 stay reserved.
No cleaning, target changes, class balancing, tuning or recalibration.
The 10% is selected inside the saved 75/25 experiment's holdout, so neither
model has trained on it. Re-score the saved 75% model on that same 10%.
The existing 75/25 assignments, weights and reports are preserved.
Report: results/logistic_regression/year_stratified_90_10.md
Compatible split assignments, model weights and evaluations are reused.
Optional: --prepared-run ID --name year_stratified_90_10_v2 --threads 1
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
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

if __package__:
    from .cohort_audit import POLICY, ROOT, select_baseline
    from .evaluation import number
    from .preprocessing import atomic_json, fit_preprocessing, literal, projection
    from .train import digest, save_weights
    from .walk_forward import create_source, source_files, training_matrix, score_fold
else:
    from cohort_audit import POLICY, ROOT, select_baseline
    from evaluation import number
    from preprocessing import atomic_json, fit_preprocessing, literal, projection
    from train import digest, save_weights
    from walk_forward import create_source, source_files, training_matrix, score_fold


YEARS = tuple(range(2012, 2018))
MODEL_DATE = "2020-01-01"
OOT_YEAR = 2020
HOLDOUT = 0.10
REFERENCE_HOLDOUT = 0.25
VERSION = 2


def pool_filter():
    return ("observation_month >= DATE '2012-01-01' AND observation_month < DATE '2018-01-01' "
            f"AND candidate_default_12m IN (0, 1) AND assumed_label_available_month < DATE '{MODEL_DATE}'")


def split_indices(targets, seed, previous_holdout=None):
    """Keep 10% inside the earlier 25%; add the remaining 15% to training."""
    targets = np.asarray(targets)
    if targets.ndim != 1 or not np.isin(targets, [0, 1]).all():
        raise ValueError("A yearly split requires known binary targets")
    classes, counts = np.unique(targets, return_counts=True)
    if classes.tolist() != [0, 1] or counts.min() < 2:
        raise ValueError("Each year needs enough loans in both classes for stratified splitting")
    indexes = np.arange(len(targets), dtype=np.int32)
    if previous_holdout is None:
        # Used by synthetic checks. Actual project runs load the saved IDs.
        _, old_holdout = train_test_split(indexes, test_size=REFERENCE_HOLDOUT,
                                         stratify=targets, random_state=seed)
        previous_holdout = np.zeros(len(targets), dtype=bool)
        previous_holdout[old_holdout] = True
    previous_holdout = np.asarray(previous_holdout, dtype=bool)
    if previous_holdout.shape != targets.shape:
        raise ValueError("Previous holdout mask does not match the yearly population")
    old_holdout = indexes[previous_holdout]
    size = int(np.ceil(len(targets) * HOLDOUT))
    holdout_classes, holdout_counts = np.unique(targets[old_holdout], return_counts=True)
    if (len(old_holdout) <= size or holdout_classes.tolist() != [0, 1]
            or holdout_counts.min() < 2):
        raise ValueError("The previous holdout needs enough events/non-events for a nested 10% split")
    added_training, holdout = train_test_split(old_holdout, test_size=size,
        stratify=targets[old_holdout], random_state=seed)
    training = np.concatenate((indexes[~previous_holdout], added_training))
    return training, holdout


def assignments(connection, folder, state, seed, save, reference=None):
    paths = []
    for year in YEARS:
        path = folder / f"split_{year}.parquet"
        saved = state["splits"].get(str(year))
        if saved is not None:
            if (not path.is_file() or digest(path) != saved["sha256"]
                    or pq.read_metadata(path).num_rows != saved["eligible"]):
                raise ValueError(f"Saved {year} assignments are missing/changed; use a new experiment name")
            print(f"Reusing {year} split assignments.", flush=True)
        else:
            if path.exists():
                raise ValueError(f"Uncheckpointed split file: {path}; use a new experiment name")
            print(f"Splitting eligible {year} loans: 90% training / 10% common holdout ...", flush=True)
            # Only IDs and labels for one year enter memory. Sort IDs so sampling
            # is reproducible even when Parquet row/batch order differs.
            query = ("SELECT loan_id, candidate_default_12m FROM source WHERE " + pool_filter()
                     + f" AND year(observation_month) = {year} ORDER BY loan_id")
            with connection.sql(query).to_arrow_reader(batch_size=65_536) as reader:
                table = reader.read_all()
            targets = table["candidate_default_12m"].to_numpy(zero_copy_only=False)
            previous_holdout = None
            if reference is not None:
                earlier, reference_folder = reference
                old_path = reference_folder / f"split_{year}.parquet"
                old_stats = earlier["splits"][str(year)]
                if digest(old_path) != old_stats["sha256"]:
                    raise ValueError(f"Changed 75/25 assignments for {year}")
                old_table = pq.read_table(old_path, columns=["loan_id", "is_holdout"])
                if not table["loan_id"].equals(old_table["loan_id"]):
                    raise ValueError(f"The eligible loan IDs/order differ from the saved 75/25 split for {year}")
                previous_holdout = old_table["is_holdout"].to_numpy(zero_copy_only=False)
                if (int(previous_holdout.sum()) != old_stats["holdout"]
                        or int(targets[previous_holdout].sum()) != old_stats["holdout_events"]):
                    raise ValueError(f"The saved 75/25 labels/membership differ for {year}")
                del old_table
            train, holdout = split_indices(targets, seed, previous_holdout)
            mask = np.zeros(table.num_rows, dtype=bool)
            mask[holdout] = True
            output = pa.table({"loan_id": table["loan_id"], "is_holdout": pa.array(mask)})
            available = connection.execute(
                f"SELECT count(*) FROM source WHERE year(observation_month) = {year}"
            ).fetchall()[0][0]
            record = {"available": available, "eligible": table.num_rows,
                      "training": len(train), "holdout": len(holdout),
                      "training_events": int(targets[train].sum()), "holdout_events": int(targets[holdout].sum())}
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                pq.write_table(output, temporary, compression="zstd")
                temporary.rename(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            record["sha256"] = digest(path)
            state["splits"][str(year)] = record
            save()
            del table, targets, train, holdout, mask, output
        paths.append(path)
    files = "[" + ", ".join(literal(path.as_posix()) for path in paths) + "]"
    connection.execute(f"CREATE TEMP VIEW assignments AS SELECT * FROM read_parquet({files})")
    connection.execute("""
        CREATE TEMP VIEW population AS
        SELECT s.*, CASE WHEN a.is_holdout THEN 'validation' ELSE 'development' END AS proposed_split
        FROM source s JOIN assignments a USING (loan_id)
        WHERE """ + pool_filter())


def counts(connection):
    result = connection.execute("""
        SELECT count(*) AS rows, sum(candidate_default_12m) AS events,
            min(observation_month) AS first_observation, max(observation_month) AS last_observation,
            max(assumed_label_available_month) AS last_label_available
        FROM eligible
    """)
    # Finish the DB-API result before switching back to relation/Arrow queries.
    record = dict(zip([column[0] for column in result.description], result.fetchall()[0]))
    if not record["rows"]:
        raise ValueError("No eligible loans in this evaluation/training group")
    record["events"] = int(record["events"])
    record["event_rate"] = record["events"] / record["rows"]
    for name in ("first_observation", "last_observation", "last_label_available"):
        record[name] = str(record[name])
    return record


def write_report(path, state):
    lines = ["# Logistic regression: yearly 90/10 split and 2020 evaluation", "",
             f"Status: {state['status']}", f"Preparation run: {state['prepared_run']}", "",
             "## Design", "",
             "- One model learns from the combined 90% training subsets of 2012-2017. The other 10% in EACH year is held out.",
             "- The new 10% is a stratified subset of the old 25% holdout. All of the old 75% training loans remain in training; the other 15% is added. Neither model trained on the common 10%.",
             f"- Holdout fraction: {100 * state['signature']['holdout_fraction']:.0f}%; fixed random seed: {state['signature']['seed']}. Every training loan has equal weight, so larger years contribute more loans.",
             "- Years mean first monthly observation, not the annual archive filename. Sampling is stratified by outcome separately within each year, with a fixed seed; it does not balance default/non-default classes.",
             "- Only known 0/1 outcomes available before January 2020 enter the 2012-2017 pool. Existing global split flags do not determine this experiment's membership.",
             "- Medians, scales and categories are refitted on the combined 90% training rows only. Held-out and 2020 rows never fit preprocessing or coefficients. The 75% reference keeps its original fitted preprocessing.",
             "- Earlier-year holdout results are retrospective same-period checks. A model trained through 2017 could not have been deployed in 2012.",
             "- 2020 is the later-period evaluation. Training outcomes are available before it starts. The unused 2018-2019 observation years separate the sample periods.",
             "- This experiment releases 2020 from the previously reserved period. Once its results guide development, 2020 is no longer an untouched final test. 2021-2023 remain reserved and are not scored.",
             "- Uses the existing L2 logistic specification and repayment-label policy, without tuning, class weights, sampling seeds selected for good results, or recalibration.", "",
             "Academic basis: same-period out-of-sample and later out-of-time evaluation are both described in [Mushava and Murray (2024), section 4.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11220865/). Split fractions and dates are this project's design choices.", "",
             "## Yearly split counts", "",
             "Available counts are candidate loans first observed that year. Eligible excludes unknown outcomes and outcomes not available by the model date.", "",
             "| Year | Available | Eligible | Training (about 90%) | Holdout (about 10%) |",
             "|---|---:|---:|---:|---:|"]
    for year, item in sorted(state["splits"].items()):
        lines.append(f"| {year} | {item['available']:,} | {item['eligible']:,} | {item['training']:,} | {item['holdout']:,} |")
    if state.get("training"):
        t = state["training"]
        lines += ["", "## One fitted model", "",
                  f"Training loans: {t['rows']:,}; bad events: {t['events']:,}; event rate: {number(t['event_rate'], True)}.",
                  f"Training observations: {t['first_observation']} to {t['last_observation']}. Latest training outcome availability: {t['last_label_available']}.",
                  f"Encoded features: {t['features']}; fitting time: {t['fit_seconds']:.1f} seconds; converged: {t['converged']}."]
    lines += ["", "## Evaluation results", "",
              "The same frozen model predicts every row below. Earlier rows use only held-out loans; 2020 uses all eligible 2020 observations.", "",
              "| Group | Loans | Bad events | ROC AUC | Mean predicted risk | Observed risk | Observed / expected | Top 10% event capture |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, item in sorted(state["evaluations"].items()):
        m = item["metrics"]
        label = f"{item['year']} holdout" if key.startswith("holdout_") else "2020 later-period"
        lines.append(f"| {label} | {m['loans']:,} | {m['bad_events']:,} | {number(m['roc_auc'])} | "
                     f"{number(m['mean_predicted_pd'], True)} | {number(m['observed_event_rate'], True)} | "
                     f"{number(m['observed_expected_ratio'])} | {number(m['top_10pct_event_capture'], True)} |")
    lines += ["", "## Probability errors", "",
              "Lower is better. The constant benchmark assigns the training event rate to every evaluated loan.", "",
              "| Group | Log loss / constant | Brier / constant |", "|---|---:|---:|"]
    for key, item in sorted(state["evaluations"].items()):
        m = item["metrics"]
        label = f"{item['year']} holdout" if key.startswith("holdout_") else "2020 later-period"
        lines.append(f"| {label} | {number(m['log_loss'])} / {number(m['constant_benchmark_log_loss'])} | "
                     f"{number(m['brier_score'])} / {number(m['constant_benchmark_brier'])} |")
    lines += ["", "## 75% versus 90% training: same evaluation loans", "",
              "The 75% reference is scored on the SAME 10% as the new model. Its earlier full-25% scores are not used for these comparisons. Both models use the same eligible 2020 evaluation population.", "",
              "| Group | Loans | AUC: 75% / 90% | Predicted risk: 75% / 90% | Observed risk | Log loss: 75% / 90% |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, item in sorted(state["evaluations"].items()):
        if "reference" not in item:
            continue
        old, new = item["reference"]["metrics"], item["metrics"]
        label = f"{item['year']} common holdout" if key.startswith("holdout_") else "2020 later-period"
        lines.append(f"| {label} | {new['loans']:,} | {number(old['roc_auc'])} / {number(new['roc_auc'])} | "
                     f"{number(old['mean_predicted_pd'], True)} / {number(new['mean_predicted_pd'], True)} | "
                     f"{number(new['observed_event_rate'], True)} | {number(old['log_loss'])} / {number(new['log_loss'])} |")
    lines += ["", "## Interpretation limits", "",
              "- ROC AUC measures ranking, not percentage correct. Observed / expected above 1 means underprediction.",
              "- Random holdouts share broad economic conditions with training. They complement later-period checks; they do not establish future stability.",
              "- A pooled model may underpredict some training-era years and overpredict others when their event rates differ.",
              "- Stratifying preserves each year's natural event proportion approximately, with rounding to whole loans. Year sizes need not be equal.",
              "- Outcome availability uses the existing assumed reporting lag, not reconstructed historical release vintages.",
              "- Earlier baseline and walk-forward results remain separate experiments. This report does not replace them or claim model approval.",
              "- The smaller holdout reduces evaluation precision, especially in small risk groups. Comparisons are diagnostic; these data have already been inspected.",
              "- Completed compatible assignments, weights and evaluations are reused. Per-year risk bands and fitted preprocessing are retained in the run's metrics.json.", ""]
    if state.get("oot_counts"):
        item = state["oot_counts"]
        lines += [f"2020 candidate observations: {item['available']:,}; evaluated: {item['eligible']:,}; unknown outcome/missing availability excluded: {item['available'] - item['eligible']:,}.", ""]
    if state.get("error"):
        lines += ["## Error", "", state["error"].replace("\n", " "), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(prepared_run=None, name="year_stratified_90_10", seed=42, threads=2, memory_gb=6.0,
         reference_name="year_stratified"):
    if not re.fullmatch(r"year_stratified(?:_[a-z0-9_]{1,32})?", name):
        raise ValueError("Use a name such as year_stratified_90_10")
    if (not re.fullmatch(r"year_stratified(?:_[a-z0-9_]{1,32})?", reference_name)
            or name == reference_name):
        raise ValueError("Select a valid, separate 75/25 reference experiment")
    if not 0 <= seed < 2**32 or not np.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("Use a non-negative 32-bit seed and a positive sparse allocation budget")
    inputs, baseline, baseline_path = select_baseline(prepared_run)
    files, fingerprints = source_files(inputs, last_year=OOT_YEAR)
    config = {key: baseline["signature"][key] for key in ("C", "solver", "max_iter", "tolerance", "class_weight")}
    if config["solver"] != "lbfgs" or config["class_weight"] is not None:
        raise ValueError("Expected the existing unweighted L2/L-BFGS baseline specification")
    run_id = inputs["prepared_run"]
    folder = ROOT / "results" / "logistic_regression" / run_id / POLICY / name
    split_folder = ROOT / "data" / "model_inputs" / "logistic_regression" / run_id / POLICY / name
    model_folder = ROOT / "data" / "models" / "logistic_regression" / run_id / POLICY / name
    checkpoint, weights = folder / "metrics.json", model_folder / "model_weights.npz"
    report = ROOT / "results" / "logistic_regression" / f"{name}.md"
    signature = {"version": VERSION, "years": list(YEARS), "oot_year": OOT_YEAR,
                 "model_date": MODEL_DATE, "holdout_fraction": HOLDOUT, "seed": seed,
                 "input_manifest_sha256": baseline["signature"]["input_manifest_sha256"],
                 "files": fingerprints, "config": config, "threads": threads,
                 "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                 "duckdb": duckdb.__version__, "pyarrow": pa.__version__}
    reference_path = folder.parent / reference_name / "metrics.json"
    reference_weights = model_folder.parent / reference_name / "model_weights.npz"
    reference_splits = split_folder.parent / reference_name
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (not reference.get("status", "").startswith("COMPLETED")
            or reference["signature"].get("holdout_fraction") != REFERENCE_HOLDOUT
            or not reference.get("training", {}).get("converged")):
        raise ValueError("The saved 75/25 reference must be complete and converged")
    for key in ("years", "oot_year", "model_date", "seed", "input_manifest_sha256", "files", "config"):
        if reference["signature"].get(key) != signature[key]:
            raise ValueError(f"The 75/25 reference uses different {key}; cannot compare on the same sample")
    if digest(reference_weights) != reference["model_sha256"]:
        raise ValueError("The saved 75% model weights changed")
    reference_expressions, reference_features = projection(reference["fitted"])
    with np.load(reference_weights, allow_pickle=False) as fitted:
        reference_coefficients = fitted["coefficients"].copy()
        reference_intercept = float(fitted["intercept"].item())
        if (fitted["feature_names"].tolist() != reference_features
                or reference_features != reference["feature_names"]
                or reference_coefficients.shape != (len(reference_features),)
                or not np.isfinite(reference_coefficients).all() or not np.isfinite(reference_intercept)):
            raise ValueError("Invalid 75% reference coefficients/preprocessing")
    signature["reference"] = {"name": reference_name, "metrics_sha256": digest(reference_path),
                              "model_sha256": reference["model_sha256"]}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Inputs/settings changed. Use --name year_stratified_90_10_v2 to preserve earlier work.")
    else:
        state = {"prepared_run": run_id, "signature": signature, "splits": {}, "evaluations": {},
                 "baseline_source": baseline_path.relative_to(ROOT).as_posix()}

    def verify_sources():
        for record in fingerprints:
            info = Path(record["path"]).stat()
            if (info.st_size, info.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise ValueError("A corrected candidate source changed during the experiment")
        if digest(baseline_path.parent.parent / "manifest.json") != signature["input_manifest_sha256"]:
            raise ValueError("Preprocessing manifest changed during the experiment")
        if (digest(reference_path) != signature["reference"]["metrics_sha256"]
                or digest(reference_weights) != signature["reference"]["model_sha256"]):
            raise ValueError("The 75/25 reference changed during comparison")

    def save():
        atomic_json(checkpoint, state)
        write_report(report, state)

    state.pop("error", None)
    state["status"] = "RUNNING - 2020 IS EVALUATION DATA; 2021-2023 RESERVED"
    save()
    try:
        pa.set_cpu_count(threads)
        with duckdb.connect(config={"memory_limit": "2GB", "threads": str(threads)}) as connection, threadpool_limits(limits=threads):
            connection.execute("SET enable_progress_bar = false")
            create_source(connection, files, cutoff="2021-01-01")
            assignments(connection, split_folder, state, seed, save, (reference, reference_splits))
            connection.execute("CREATE TEMP VIEW eligible AS SELECT * FROM population WHERE proposed_split = 'development'")
            training_counts = counts(connection)
            expected_rows = sum(item["training"] for item in state["splits"].values())
            expected_events = sum(item["training_events"] for item in state["splits"].values())
            if (training_counts["rows"] != expected_rows or training_counts["events"] != expected_events
                    or training_counts["last_observation"] >= "2018-01-01"
                    or training_counts["last_label_available"] >= MODEL_DATE):
                raise ValueError("Training membership/dates differ from the saved yearly assignments")
            if weights.exists():
                if not state.get("model_sha256") or digest(weights) != state["model_sha256"]:
                    raise ValueError("Uncheckpointed/changed weights; use a new experiment name")
                expressions, features = projection(state["fitted"])
                with np.load(weights, allow_pickle=False) as fitted:
                    coefficients = fitted["coefficients"].copy()
                    intercept = float(fitted["intercept"].item())
                    if fitted["feature_names"].tolist() != features or features != state["feature_names"]:
                        raise ValueError("Saved model and preprocessing feature order differ")
                if any(state["training"][key] != value for key, value in training_counts.items()):
                    raise ValueError("Saved training statistics differ from current membership")
                print("Reusing fitted model; no training repeated.", flush=True)
            else:
                if state.get("training"):
                    raise ValueError("Saved model weights are missing; use a new experiment name")
                print("Learning preprocessing from the pooled 90% training subsets only ...", flush=True)
                state["fitted"] = fit_preprocessing(connection)
                expressions, features = projection(state["fitted"])
                print(f"Loading {expected_rows:,} training loans from 2012-2017 ...", flush=True)
                matrix, labels = training_matrix(connection, expressions, features, expected_rows, OOT_YEAR, memory_gb)
                estimator = LogisticRegression(C=config["C"], solver="lbfgs", l1_ratio=0.0,
                    max_iter=config["max_iter"], tol=config["tolerance"], class_weight=None)
                print("Fitting one logistic-regression model ...", flush=True)
                started = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(matrix, labels)
                training = dict(training_counts)
                training.update(fit_seconds=time.monotonic() - started, features=len(features),
                                iterations=int(estimator.n_iter_.max()))
                training["converged"] = training["iterations"] < config["max_iter"] and not any(
                    issubclass(item.category, ConvergenceWarning) for item in caught)
                if estimator.classes_.tolist() != [0, 1]:
                    raise ValueError("Unexpected fitted outcome classes")
                coefficients, intercept = estimator.coef_[0].copy(), float(estimator.intercept_[0])
                del matrix, labels, estimator
                if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
                    raise ValueError("Non-finite fitted coefficients")
                verify_sources()
                model_folder.mkdir(parents=True, exist_ok=True)
                save_weights(weights, coefficients, intercept, features)
                state.update(training=training, feature_names=features, model_sha256=digest(weights),
                             warnings=[str(item.message) for item in caught])
                save()
            if coefficients.shape != (len(features),) or not np.isfinite(coefficients).all() or not np.isfinite(intercept):
                raise ValueError("Invalid fitted model coefficients")
            for year in (*YEARS, OOT_YEAR):
                key = f"holdout_{year}" if year in YEARS else f"oot_{year}"
                if year in YEARS:
                    connection.execute(f"CREATE OR REPLACE TEMP VIEW eligible AS SELECT * FROM population "
                                       f"WHERE proposed_split = 'validation' AND year(observation_month) = {year}")
                else:
                    connection.execute(f"""
                        CREATE OR REPLACE TEMP VIEW eligible AS
                        SELECT *, 'validation' AS proposed_split FROM source
                        WHERE observation_month >= DATE '{year}-01-01' AND observation_month < DATE '{year + 1}-01-01'
                            AND candidate_default_12m IN (0, 1) AND assumed_label_available_month IS NOT NULL
                    """)
                current = counts(connection)
                if year in YEARS:
                    saved = state["splits"][str(year)]
                    if current["rows"] != saved["holdout"] or current["events"] != saved["holdout_events"]:
                        raise ValueError(f"Holdout {year} membership differs from saved assignments")
                else:
                    available = connection.execute(f"SELECT count(*) FROM source WHERE year(observation_month) = {year}").fetchall()[0][0]
                    state["oot_counts"] = {"available": available, "eligible": current["rows"]}
                if key in state["evaluations"]:
                    if state["evaluations"][key]["counts"] != current:
                        raise ValueError(f"Saved {key} counts changed")
                    print(f"Reusing completed {key} evaluation.", flush=True)
                    continue
                print(f"Scoring {current['rows']:,} loans: {key} ...", flush=True)
                result = score_fold(connection, expressions, features, coefficients, intercept,
                                    current["rows"], year, state["training"]["event_rate"])
                old_evaluation = reference["evaluations"].get(key)
                if year == OOT_YEAR and old_evaluation and old_evaluation["counts"] == current:
                    reference_result = {field: old_evaluation[field] for field in ("metrics", "bands")}
                    print("Reusing the 75% model's evaluation on the same 2020 loans.", flush=True)
                else:
                    print(f"Scoring the saved 75% model on the same {current['rows']:,} held-out loans ...", flush=True)
                    reference_result = score_fold(connection, reference_expressions, reference_features,
                        reference_coefficients, reference_intercept, current["rows"], year,
                        reference["training"]["event_rate"])
                if (reference_result["metrics"]["loans"] != result["metrics"]["loans"]
                        or reference_result["metrics"]["bad_events"] != result["metrics"]["bad_events"]):
                    raise ValueError("The 75% and 90% evaluations do not describe the same loans/outcomes")
                verify_sources()
                state["evaluations"][key] = {**result, "reference": reference_result, "year": year, "counts": current}
                save()
            verify_sources()
        state["status"] = ("COMPLETED - REVIEW SAME-PERIOD AND 2020 EVALUATION"
            if state["training"]["converged"] else "COMPLETED WITH NONCONVERGENCE - REVIEW REQUIRED")
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    except Exception as error:
        state.update(status="FAILED - SAVED PROGRESS RETAINED", error=str(error))
        save()
        raise
    print(f"Finished. OPEN THIS REPORT: {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run")
    parser.add_argument("--name", default="year_stratified_90_10")
    parser.add_argument("--reference-name", default="year_stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--memory-gb", type=float, default=6.0)
    args = parser.parse_args()
    main(args.prepared_run, args.name, args.seed, args.threads, args.memory_gb, args.reference_name)
