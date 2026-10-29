"""Train the first logistic baseline and report VALIDATION performance only.

Run normally in PyCharm after preprocessing.py finishes. No parameters needed.
Uses all eligible development loans, then scores validation loans. Never fits,
tunes, calibrates or evaluates on the final test split. No class balancing.

Reads corrected verified_repayment_v2 inputs. Original datasets are read-only.
Uses bounded Arrow batches and a preallocated sparse training matrix, not a
large dense pandas table. Defaults: L2 logistic, C=1, L-BFGS, 200 iterations,
two compute threads. Nonconvergence is clearly reported, not silently accepted.

Results: results/logistic_regression/<preparation_run>/verified_repayment_v2/baseline/
Main report to open: results/report.md (short summary, updated on completion).
Model: data/models/logistic_regression/<preparation_run>/verified_repayment_v2/baseline/
Compatible fitted weights are reused on rerun. Use --name another_name for an
intentional new experiment, rather than overwriting earlier results.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy
from scipy.sparse import csr_matrix
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data_preparation.repayment_targets import POLICY
if __package__:
    from .evaluation import evaluate, write_report, write_summary
    from .preprocessing import atomic_json
else:
    from evaluation import evaluate, write_report, write_summary
    from preprocessing import atomic_json

VERSION = 1
METADATA = ["loan_id", "observation_month", "candidate_default_12m", "proposed_split"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_state(folder, state):
    atomic_json(folder / "metrics.json", state)
    write_report(folder / "report.md", state)


def select_inputs(prepared_run):
    root = ROOT / "results" / "logistic_regression"
    if prepared_run is not None:
        if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{6}", prepared_run):
            raise ValueError("Invalid preparation run ID")
        candidates = [root / prepared_run / POLICY / "manifest.json"]
    else:
        candidates = sorted(root.glob(f"*/{POLICY}/manifest.json"), reverse=True)
    for manifest in candidates:
        inputs = json.loads(manifest.read_text(encoding="utf-8"))
        if inputs.get("status", "").startswith("INPUTS PREPARED") and inputs.get("signature", {}).get("target_policy") == POLICY:
            break
    else:
        raise ValueError("Corrected preprocessing is not complete. Run preprocessing.py first; no training started.")
    for check in ("missing_or_duplicate_ids", "temporal_rule_violations"):
        if inputs.get("checks", {}).get(check) != 0:
            raise ValueError(f"Required preprocessing check did not pass: {check}")
    if set(inputs.get("files", {})) != set(map(str, inputs["years"])):
        raise ValueError("The preprocessing manifest does not contain every selected year")
    if set(inputs.get("label_updates", {})) != set(map(str, inputs["years"])):
        raise ValueError("Missing repayment-policy audit results")
    for update in inputs["label_updates"].values():
        if any(update["stats"].get(key) != 0 for key in
               ("lost_default_labels", "changed_known_labels", "unsupported_zero_labels")):
            raise ValueError("Repayment-policy preservation checks failed")
    features = inputs["feature_names"]
    numeric = inputs["signature"]["numeric"]
    categorical = inputs["signature"]["categorical"]
    expected = [feature for name in numeric for feature in (name + "_scaled", name + "_missing")]
    for name in categorical:
        expected += [name + "_unknown"] + [f"{name}_category_{index:02d}" for index in
                    range(1, len(inputs["fitted"]["categorical"][name]) + 1)]
    if features != expected or len(features) != len(set(features)) or set(features).intersection(METADATA):
        raise ValueError("Feature whitelist differs from the fitted preprocessing specification")
    # There are at most two nonzero values per numeric input and exactly one
    # nonzero one-hot value per categorical input. This bounds sparse storage.
    max_nonzero = 2 * len(numeric) + len(categorical)
    files, fingerprints = [], []
    for year in sorted(inputs["files"], key=int):
        record = inputs["files"][year]
        path = (ROOT / record["file"]).resolve()
        path.relative_to(ROOT.resolve())
        metadata = pq.read_metadata(path)
        info = path.stat()
        if (metadata.num_rows != record["rows"] or metadata.schema.names != METADATA + features
                or info.st_size != record["bytes"]):
            raise ValueError(f"Model input differs from its completed preprocessing report: {path}")
        files.append((path, record["splits"]))
        fingerprints.append({"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                             "rows": metadata.num_rows})
    return manifest, inputs, files, fingerprints, max_nonzero


def batches(files, split, features, batch_size):
    selected = [str(path) for path, counts in files if counts.get(split, 0) > 0]
    if not selected:
        raise ValueError(f"No eligible {split} loans in this preprocessing run")
    scanner = ds.dataset(selected, format="parquet").scanner(
        columns=METADATA + features, filter=ds.field("proposed_split") == split,
        batch_size=batch_size, batch_readahead=1, fragment_readahead=1, use_threads=True,
    )
    for batch in scanner.to_batches():
        if batch.num_rows:
            yield batch


def unpack(batch, features):
    target = batch.column(batch.schema.get_field_index("candidate_default_12m"))
    raw_y = target.to_numpy(zero_copy_only=False)
    if target.null_count or not np.isin(raw_y, [0, 1]).all():
        raise ValueError("Unknown or invalid target reached model training/evaluation")
    y = raw_y.astype(np.int8)
    dates = batch.column(batch.schema.get_field_index("observation_month")).to_numpy(
        zero_copy_only=False).astype("datetime64[D]")
    if np.isnat(dates).any():
        raise ValueError("Missing observation dates in model inputs")
    dense = np.column_stack([batch.column(batch.schema.get_field_index(name)).to_numpy(
        zero_copy_only=False) for name in features]).astype(np.float64, copy=False)
    if not np.isfinite(dense).all():
        raise ValueError("Missing/non-finite encoded predictor; rerun/review preprocessing")
    return csr_matrix(dense), y, dates


def load_training(files, features, expected_rows, max_nonzero, batch_size, memory_gb):
    capacity = expected_rows * max_nonzero
    if capacity >= np.iinfo(np.int32).max:
        raise ValueError("Sparse matrix exceeds this baseline's 32-bit indexing capacity")
    storage = capacity * 12 + (expected_rows + 1) * 4 + expected_rows
    # This is an allocation guard, not a hard process/Windows memory limit.
    if 2 * storage + 512 * 1024**2 > memory_gb * 1024**3:
        raise MemoryError("Sparse-matrix allocation guard exceeded. Review available RAM before increasing --memory-gb.")
    print(f"Loading {expected_rows:,} development loans; sparse allocation bound {storage / 1024**3:.2f} GiB ...", flush=True)
    values = np.empty(capacity, dtype=np.float64)
    indices = np.empty(capacity, dtype=np.int32)
    pointers = np.empty(expected_rows + 1, dtype=np.int32)
    pointers[0] = 0
    targets = np.empty(expected_rows, dtype=np.int8)
    offset = nonzero = 0
    first = last = None
    started = last_progress = time.monotonic()
    for batch in batches(files, "development", features, batch_size):
        matrix, y, dates = unpack(batch, features)
        end = offset + len(y)
        if end > expected_rows or np.max(np.diff(matrix.indptr)) > max_nonzero:
            raise ValueError("Training row count or one-hot sparsity assumptions were violated")
        count = matrix.nnz
        values[nonzero:nonzero + count] = matrix.data
        indices[nonzero:nonzero + count] = matrix.indices
        pointers[offset + 1:end + 1] = matrix.indptr[1:] + nonzero
        targets[offset:end] = y
        offset, nonzero = end, nonzero + count
        start_date, end_date = dates.min(), dates.max()
        first = start_date if first is None else min(first, start_date)
        last = end_date if last is None else max(last, end_date)
        now = time.monotonic()
        if now - last_progress >= 5:
            print(f"  Loaded {offset:,}/{expected_rows:,} development loans", flush=True)
            last_progress = now
    if offset != expected_rows or np.unique(targets).size != 2:
        raise ValueError("Development row count differs from the manifest, or one target class is absent")
    if last >= np.datetime64("2018-01-01"):
        raise ValueError("A validation/test observation reached development fitting")
    matrix = csr_matrix((values[:nonzero], indices[:nonzero], pointers),
                        shape=(expected_rows, len(features)), copy=False)
    stats = {"rows": expected_rows, "events": int(targets.sum()), "event_rate": float(targets.mean()),
             "first_observation": str(first), "last_observation": str(last), "features": len(features),
             "sparse_storage_bound_gib": storage / 1024**3,
             "load_seconds": time.monotonic() - started}
    return matrix, targets, stats


def save_weights(path, coefficients, intercept, features):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, coefficients=coefficients, intercept=np.array(intercept),
                                feature_names=np.asarray(features, dtype="U"))
        if path.exists():
            raise FileExistsError(path)
        temporary.rename(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def score_validation(files, features, coefficients, intercept, expected_rows, training_rate, target, batch_size):
    y_all = np.empty(expected_rows, dtype=np.int8)
    p_all = np.empty(expected_rows, dtype=np.float64)
    years = np.empty(expected_rows, dtype=np.int16)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
    offset = 0
    first = last = None
    writer = None
    last_progress = time.monotonic()
    try:
        for batch in batches(files, "validation", features, batch_size):
            matrix, y, dates = unpack(batch, features)
            if np.any((dates < np.datetime64("2018-01-01")) | (dates >= np.datetime64("2020-01-01"))):
                raise ValueError("A non-validation observation reached validation evaluation")
            p = expit(np.asarray(matrix @ coefficients).ravel() + intercept)
            end = offset + len(y)
            if end > expected_rows:
                raise ValueError("Validation row count exceeds the completed preprocessing manifest")
            y_all[offset:end], p_all[offset:end] = y, p
            years[offset:end] = dates.astype("datetime64[Y]").astype(np.int64) + 1970
            table = pa.table({
                "loan_id": batch.column(batch.schema.get_field_index("loan_id")),
                "observation_month": batch.column(batch.schema.get_field_index("observation_month")),
                "observed_bad_event_12m": pa.array(y), "predicted_pd": pa.array(p),
            })
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            offset = end
            first = dates.min() if first is None else min(first, dates.min())
            last = dates.max() if last is None else max(last, dates.max())
            now = time.monotonic()
            if now - last_progress >= 5:
                print(f"  Scored {offset:,}/{expected_rows:,} validation loans", flush=True)
                last_progress = now
        if writer is not None:
            writer.close()
            writer = None
        if offset != expected_rows or pq.read_metadata(temporary).num_rows != expected_rows:
            raise ValueError("Validation prediction row count differs from the input manifest")
        metrics, bands = evaluate(y_all, p_all, training_rate)
        by_year = {}
        for year in np.unique(years):
            selected = years == year
            by_year[str(int(year))] = evaluate(y_all[selected], p_all[selected], training_rate)[0]
        result = {"metrics": metrics, "bands": bands, "by_year": by_year,
                  "first_observation": str(first), "last_observation": str(last)}
        # Own diagnostic output only; a fitted model is never overwritten.
        temporary.replace(target)
        return result
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(prepared_run=None, name="baseline", c=1.0, max_iter=200, tolerance=1e-5,
         batch_size=65_536, memory_gb=6.0, threads=2):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", name):
        raise ValueError("Use a short lowercase experiment name")
    if (not np.isfinite(c) or c <= 0 or max_iter < 1 or not np.isfinite(tolerance)
            or tolerance <= 0 or batch_size < 1 or not np.isfinite(memory_gb) or memory_gb <= 0):
        raise ValueError("C, max_iter, tolerance, batch_size and memory budget must be positive")
    manifest, inputs, files, fingerprints, max_nonzero = select_inputs(prepared_run)
    features = inputs["feature_names"]
    counts = {split: sum(item.get(split, 0) for _, item in files) for split in ("development", "validation")}
    if min(counts.values()) == 0:
        raise ValueError("Both development and validation groups must be populated")
    run_id = inputs["prepared_run"]
    output = ROOT / "results" / "logistic_regression" / run_id / POLICY / name
    model_folder = ROOT / "data" / "models" / "logistic_regression" / run_id / POLICY / name
    output.mkdir(parents=True, exist_ok=True)
    model_folder.mkdir(parents=True, exist_ok=True)
    weights = model_folder / "model_weights.npz"
    signature = {"version": VERSION, "target_policy": POLICY, "input_manifest_sha256": digest(manifest),
                 "input_files": fingerprints, "features": features,
                 "estimator": "L2 logistic regression", "solver": "lbfgs", "C": c,
                 "max_iter": max_iter, "tolerance": tolerance, "class_weight": None,
                 "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                 "sklearn_version": sklearn.__version__}
    checkpoint = output / "metrics.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError("Existing experiment inputs/settings differ. Use --name another_name; old results will not be overwritten.")
        state.pop("error", None)
    else:
        state = {"prepared_run": run_id, "signature": signature, "warnings": []}
    if weights.exists() and ("model_sha256" not in state or digest(weights) != state["model_sha256"]):
        raise ValueError("Uncheckpointed/changed model weights; select a new experiment name rather than overwriting")
    state["status"] = "RUNNING - DEVELOPMENT FIT / VALIDATION EVALUATION ONLY"
    save_state(output, state)
    started = time.monotonic()
    try:
        pa.set_cpu_count(threads)
        with threadpool_limits(limits=threads):
            if weights.exists():
                with np.load(weights, allow_pickle=False) as fitted:
                    if fitted["feature_names"].tolist() != features:
                        raise ValueError("Saved model feature order differs from the inputs")
                    coefficients = fitted["coefficients"].copy()
                    intercept = float(fitted["intercept"].item())
                print("Reusing fitted baseline weights; development fitting is skipped.", flush=True)
            else:
                matrix, y, training = load_training(files, features, counts["development"], max_nonzero,
                                                    batch_size, memory_gb)
                print("Fitting logistic regression with L-BFGS. This may take time ...", flush=True)
                estimator = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=max_iter,
                                               tol=tolerance, class_weight=None, fit_intercept=True)
                fitting_started = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(matrix, y)
                training["fit_seconds"] = time.monotonic() - fitting_started
                training["iterations"] = int(estimator.n_iter_.max())
                convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
                training["converged"] = not convergence_warnings and training["iterations"] < max_iter
                state["warnings"] = [str(item.message) for item in caught]
                state["training"] = training
                if estimator.classes_.tolist() != [0, 1]:
                    raise ValueError("Unexpected logistic target classes")
                coefficients = estimator.coef_[0].copy()
                intercept = float(estimator.intercept_[0])
                del matrix, y, estimator
                if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
                    raise ValueError("Non-finite fitted coefficients; model weights were not saved")
                save_weights(weights, coefficients, intercept, features)
                state["model_sha256"] = digest(weights)
                state["model_file"] = str(weights.relative_to(ROOT))
                atomic_json(model_folder / "preprocessing.json", {
                    "target_policy": POLICY, "feature_names": features, "fitted": inputs["fitted"],
                    "source_manifest": str(manifest.relative_to(ROOT)),
                    "note": "Weights expect encoded inputs in exactly this feature order. No future metadata is an input.",
                })
                save_state(output, state)
                print(f"Fit completed in {training['fit_seconds']:.1f}s; converged={training['converged']}.", flush=True)
            if (coefficients.shape != (len(features),) or not np.isfinite(coefficients).all()
                    or not np.isfinite(intercept)):
                raise ValueError("Invalid fitted coefficients")
            print("Scoring validation only; final test is not evaluated ...", flush=True)
            state["validation"] = score_validation(
                files, features, coefficients, intercept, counts["validation"], state["training"]["event_rate"],
                output / "validation_predictions.parquet", batch_size,
            )
        for item in fingerprints:
            info = Path(item["path"]).stat()
            if (info.st_size, info.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise ValueError("A preprocessed input changed during training/evaluation")
        if digest(manifest) != signature["input_manifest_sha256"]:
            raise ValueError("The preprocessing manifest changed during training/evaluation")
        readable = {feature: feature for feature in features}
        for category, vocabulary in inputs["fitted"]["categorical"].items():
            for index, value in enumerate(vocabulary, start=1):
                readable[f"{category}_category_{index:02d}"] = f"{category} = {value}"
        rows = [{"feature": name, "description": readable[name], "coefficient": float(value)}
                for name, value in zip(features, coefficients)]
        rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
        write_csv(output / "coefficients.csv", rows)
        write_csv(output / "risk_bands.csv", state["validation"]["bands"])
        state["intercept"] = intercept
        state["last_execution_seconds"] = time.monotonic() - started
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["status"] = ("BASELINE FITTED - VALIDATION ONLY; NOT APPROVED" if state["training"]["converged"]
                           else "OPTIMIZER NOT CONVERGED - VALIDATION METRICS ARE PROVISIONAL")
        save_state(output, state)
        write_summary(ROOT / "results" / "report.md", state, output / "report.md")
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        save_state(output, state)
        raise
    metrics = state["validation"]["metrics"]
    print(f"Validation ROC AUC: {metrics['roc_auc']}", flush=True)
    print(f"Validation observed rate: {metrics['observed_event_rate']:.3%}; "
          f"mean predicted probability: {metrics['mean_predicted_pd']:.3%}", flush=True)
    print(f"Finished. OPEN THIS REPORT: {ROOT / 'results' / 'report.md'}", flush=True)
    print(f"Optional detailed report: {output / 'report.md'}", flush=True)
    print("Final test was not scored. No lending decisions or approval thresholds were selected.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--C", dest="c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--memory-gb", type=float, default=6.0)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    main(args.prepared_run, args.name, args.c, args.max_iter, args.tolerance,
         args.batch_size, args.memory_gb, args.threads)
