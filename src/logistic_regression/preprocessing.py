"""Prepare numeric inputs for the first model; do not train a model.

Run normally in PyCharm. Defaults to 2012-2023 and automatically locates a
completed preparation run. Optional: --prepared-run ID --years 2012 2013 ...

Reuses cleaned data. First checks five monthly columns for early-exit loans
to retain verified successful repayments as 0. Existing bad events remain 1;
sales and unresolved histories stay unknown. No normalization is repeated.
Split boundaries and reporting-lag rules remain provisional and unchanged.
Fits medians, scaling and category vocabulary using eligible development rows
only. Missing outcomes are excluded, never filled. Original files stay intact.

Outputs: data/model_inputs/logistic_regression/<preparation_run>/verified_repayment_v2/
Reports: results/logistic_regression/<preparation_run>/verified_repayment_v2/
Corrected targets: data/prepared/<preparation_run>/target_revisions/verified_repayment_v2/
Completed compatible outputs are reused. Stop other copies before running.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data_preparation.repayment_targets import POLICY, fingerprint, refine_targets

NUMERIC = ["fico", "original_balance", "original_rate", "ltv", "cltv", "dti", "term_months"]
CATEGORICAL = ["borrowers_code", "property_state", "property_type", "occupancy", "purpose", "channel"]
KEYS = ["loan_id", "observation_month", "candidate_default_12m", "proposed_split"]
REQUIRED = KEYS + NUMERIC + CATEGORICAL + [
    "proposed_split_eligible", "assumed_label_available_month", "research_rules_require_review",
]
VERSION = 2


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def atomic_json(path, value):
    temporary = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".checkpoint_", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def select_run(years, requested):
    folder = ROOT / "results" / "data_preparation"
    if requested is not None:
        if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{6}", requested):
            raise ValueError("Invalid preparation run ID")
        candidates = [folder / requested / "manifest.json"]
    else:
        candidates = sorted(folder.glob("*/manifest.json"), reverse=True)
    for path in candidates:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved.get("status", "").startswith("PREPARED")
                and set(years).issubset(saved.get("years", []))
                and saved.get("run_id") == path.parent.name
                and all("cohort" in saved.get("results", {}).get(str(year), {}) for year in years)):
            return saved
    raise ValueError("No completed preparation run covers the selected years. Run preparation first.")


def report(path, state):
    lines = ["# Logistic-regression input preparation", "",
             f"Status: {state['status']}", f"Preparation run: {state['prepared_run']}",
             "Years: " + ", ".join(map(str, state["years"])), "",
             "No logistic model has been fitted. Research labels/splits remain provisional.", "",
             "## Rules", "",
             "- Source/cleaned Parquets and earlier results are not modified. Normalization is not repeated.",
             "- The first corrected-target run reads five monthly columns for early-exit evidence; compatible corrected targets are then reused.",
             "- Unknown targets and split-ineligible rows are excluded, never imputed.",
             "- Verified code-01 full repayments with adequate history and no earlier bad event receive 0.",
             "- Existing bad-event labels remain 1 even after later payments; other exits and uncertain histories stay unknown.",
             "- Chronological boundaries are unchanged. Outcome availability cannot precede the original horizon/confirmation plus lag.",
             "- Medians, population means/standard deviations and category vocabulary use development rows only.",
             "- Numeric missing values use the training median; every numeric input has a separate missing flag.",
             "- Numeric values are standardized after imputation; constant columns use a scale of one.",
             "- Missing/unseen categories use an explicit unknown indicator. Categories are one-hot encoded.",
             "- Borrower count is grouped as single/multiple for consistent meaning across the 2018 disclosure change.",
             "- IDs, dates, split names and outcomes are retained as metadata, NOT predictive inputs.",
             "- Reused outputs are checked by metadata, not exhaustive value comparison.", ""]
    if state.get("label_updates"):
        lines += ["## Repayment-label update", "",
                  "Original candidate tables remain unchanged. New target versions retain all original rows and the earlier labels/reasons for audit.", "",
                  "| Year | Verified repayments changed to 0 | Of these, split eligible | Bad events before | Bad events after | Early exits still unknown |",
                  "|---|---:|---:|---:|---:|---:|"]
        for year, update in sorted(state["label_updates"].items()):
            counts = update["stats"]
            lines.append(f"| {year} | {counts['repaid_changed_to_zero']:,} | "
                         f"{counts['repaid_now_split_eligible']:,} | {counts['defaults_before']:,} | "
                         f"{counts['defaults_after']:,} | {counts['early_exits_still_unknown']:,} |")
        lines += ["", "A verified payoff needs continuous, non-duplicated reporting through repayment, known pre-payoff statuses, and no conflicting earlier exit/bad event. A blank terminal status is allowed only with the verified code-01 payoff itself.", "",
                  "Repayment markers, old labels and payoff dates are audit information, never predictors.", ""]
    if "checks" in state:
        lines += ["## Input checks", "", "| Measure | Value |", "|---|---:|"]
        lines += [f"| {key} | {value:,} |" for key, value in state["checks"].items()]
        lines += [""]
    if "fitted" in state:
        lines += ["## Numeric preprocessing: development only", "",
                  "| Input | Missing training values | Median | Mean after filling | Scale |",
                  "|---|---:|---:|---:|---:|"]
        for name, values in state["fitted"]["numeric"].items():
            lines.append(f"| {name} | {values['missing']:,} | {values['median']:.6g} | "
                         f"{values['mean']:.6g} | {values['scale']:.6g} |")
        lines += ["", "Training category lists and exact output names are saved in manifest.json.", ""]
    lines += ["## Output files", "", "| Year | Development rows | Validation rows | Test rows |", "|---|---:|---:|---:|"]
    for year, values in sorted(state.get("files", {}).items()):
        counts = values["splits"]
        lines.append(f"| {year} | {counts.get('development', 0):,} | "
                     f"{counts.get('validation', 0):,} | {counts.get('test', 0):,} |")
    if state.get("error"):
        lines += ["", "## Error", "", state["error"].replace("\n", " ")]
    lines += ["", "The agreed voluntary-payoff rule is applied. Temporal exclusions and other provisional research rules still require review before model fitting.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def save(report_root, state):
    atomic_json(report_root / "manifest.json", state)
    report(report_root / "report.md", state)


def fit_preprocessing(connection):
    """Use development rows only; no validation/test outcomes or inputs enter fitting."""
    aggregates = []
    for name in NUMERIC:
        aggregates += [f"median({identifier(name)}) AS {identifier(name + '_median')}",
                       f"count(*) FILTER (WHERE {identifier(name)} IS NULL) AS {identifier(name + '_missing')}"]
    result = connection.execute("SELECT " + ", ".join(aggregates)
                                + " FROM eligible WHERE proposed_split = 'development'")
    values = dict(zip([column[0] for column in result.description], result.fetchone()))
    fitted = {"numeric": {}, "categorical": {}}
    statistics = []
    for name in NUMERIC:
        median = values[name + "_median"]
        if median is None:
            raise ValueError(f"No observed training values for {name}; review this feature rather than inventing a median")
        filled = f"coalesce({identifier(name)}, {float(median)!r})"
        statistics += [f"avg({filled}) AS {identifier(name + '_mean')}",
                       f"stddev_pop({filled}) AS {identifier(name + '_scale')}"]
        fitted["numeric"][name] = {"median": float(median), "missing": values[name + "_missing"]}
    result = connection.execute("SELECT " + ", ".join(statistics)
                                + " FROM eligible WHERE proposed_split = 'development'")
    scales = dict(zip([column[0] for column in result.description], result.fetchone()))
    for name in NUMERIC:
        fitted["numeric"][name].update(mean=float(scales[name + "_mean"]),
                                       scale=float(scales[name + "_scale"] or 1.0))
    for name in CATEGORICAL:
        rows = connection.execute(
            f"SELECT DISTINCT {identifier(name)} FROM eligible WHERE proposed_split = 'development' "
            f"AND {identifier(name)} IS NOT NULL ORDER BY 1"
        ).fetchall()
        fitted["categorical"][name] = [row[0] for row in rows]
    return fitted


def projection(fitted):
    expressions = [identifier(name) for name in KEYS]
    names = []
    for name in NUMERIC:
        values = fitted["numeric"][name]
        output = name + "_scaled"
        expressions.append(f"CAST((coalesce({identifier(name)}, {values['median']!r}) "
                           f"- {values['mean']!r}) / {values['scale']!r} AS FLOAT) AS {identifier(output)}")
        expressions.append(f"CAST({identifier(name)} IS NULL AS UTINYINT) AS {identifier(name + '_missing')}")
        names.extend([output, name + "_missing"])
    for name in CATEGORICAL:
        categories = fitted["categorical"][name]
        known = ", ".join(map(literal, categories))
        condition = f"{identifier(name)} IS NULL"
        if categories:
            condition += f" OR {identifier(name)} NOT IN ({known})"
        output = name + "_unknown"
        expressions.append(f"CAST(({condition}) AS UTINYINT) AS {identifier(output)}")
        names.append(output)
        for index, category in enumerate(categories, start=1):
            output = f"{name}_category_{index:02d}"
            expressions.append(f"CAST(coalesce({identifier(name)} = {literal(category)}, FALSE) "
                               f"AS UTINYINT) AS {identifier(output)}")
            names.append(output)
    return expressions, names


def main(years, prepared_run=None, memory_limit="4GB", threads=2):
    years = sorted(set(years))
    if not years or any(year < 2012 or year > 2025 for year in years):
        raise ValueError("Select full years from 2012 to 2025")
    prepared = select_run(years, prepared_run)
    run_id = prepared["run_id"]
    output = ROOT / "data" / "model_inputs" / "logistic_regression" / run_id / POLICY
    report_root = ROOT / "results" / "logistic_regression" / run_id / POLICY
    target_root = ROOT / "data" / "prepared" / run_id / "target_revisions" / POLICY
    target_reports = ROOT / "results" / "data_preparation" / run_id / "target_revisions" / POLICY
    sources, fingerprints = [], []
    performance_sources = {}
    for year in years:
        source = ROOT / "data" / "prepared" / run_id / str(year) / "cohort_candidates.parquet"
        metadata = pq.read_metadata(source)
        if not set(REQUIRED).issubset(metadata.schema.names):
            raise ValueError(f"Missing required model input fields: {source}")
        if metadata.num_rows != prepared["results"][str(year)]["cohort"]["metrics"]["loan_ids"]:
            raise ValueError(f"Prepared cohort row count differs from its report: {source}")
        info = source.stat()
        sources.append(source)
        fingerprints.append({"path": str(source.relative_to(ROOT)), "bytes": info.st_size,
                             "mtime_ns": info.st_mtime_ns, "rows": metadata.num_rows})
        performance_sources[year] = []
        cached_files = {row["file"]: row for row in prepared["results"][str(year)]["files"]}
        for quarter in range(1, 5):
            path = source.parent / f"perf_{year}Q{quarter}.parquet"
            metadata = pq.read_metadata(path)
            cached = cached_files.get(path.name)
            if (cached is None or metadata.num_rows != cached["rows_after"]
                    or path.stat().st_size != cached["bytes_after"]):
                raise ValueError(f"Cleaned performance file differs from its preparation report: {path}")
            performance_sources[year].append(path)
    base_sources = list(sources)
    performance_fingerprints = [fingerprint(path) for paths in performance_sources.values() for path in paths]
    signature = {"version": VERSION, "years": years, "sources": fingerprints,
                 "numeric": NUMERIC, "categorical": CATEGORICAL,
                 "target_policy": POLICY, "reporting_lag_months": prepared["lag_months"],
                 "performance_sources": performance_fingerprints,
                 "duckdb_version": duckdb.__version__}
    manifest = report_root / "manifest.json"
    if manifest.exists():
        state = json.loads(manifest.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Inputs/settings changed. Existing model inputs will not be overwritten; review the saved run first.")
        state.pop("error", None)
    else:
        state = {"prepared_run": run_id, "years": years, "signature": signature, "files": {}}
    output.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state["status"] = "RUNNING - PROVISIONAL RESEARCH RULES"
    save(report_root, state)
    try:
        with duckdb.connect(config={
            "memory_limit": memory_limit, "threads": str(threads),
            "temp_directory": (output / "_spill").as_posix(), "max_temp_directory_size": "20GB",
            "preserve_insertion_order": "false",
        }) as connection:
            corrected_sources = []
            for year, source in zip(years, base_sources):
                target = target_root / str(year) / "cohort_candidates.parquet"
                update = refine_targets(connection, source, performance_sources[year], target,
                                        target_reports / f"{year}.json", prepared["lag_months"])
                state.setdefault("label_updates", {})[str(year)] = update
                corrected_sources.append(target)
                save(report_root, state)
            sources = corrected_sources
            paths = "[" + ", ".join(literal(path.as_posix()) for path in sources) + "]"
            connection.execute(f"CREATE TEMP VIEW source AS SELECT * FROM read_parquet({paths}, filename=true)")
            print("Checking IDs across selected years ...", flush=True)
            bad_ids = connection.execute(
                "SELECT count(*) FROM (SELECT loan_id FROM source GROUP BY loan_id "
                "HAVING loan_id IS NULL OR count(*) != 1)"
            ).fetchone()[0]
            if bad_ids:
                raise ValueError(f"Found {bad_ids:,} missing/duplicate IDs across selected cohorts; review before fitting")
            category_projection = []
            for name in CATEGORICAL:
                value = ("CASE WHEN borrowers_code = 1 THEN 'single' "
                         "WHEN borrowers_code >= 2 THEN 'multiple' END" if name == "borrowers_code"
                         else f"nullif(trim(CAST({identifier(name)} AS VARCHAR)), '')")
                category_projection.append(f"{value} AS {identifier(name)}")
            numeric_projection = [f"CASE WHEN isfinite({identifier(name)}) THEN "
                                  f"CAST({identifier(name)} AS DOUBLE) END AS {identifier(name)}" for name in NUMERIC]
            connection.execute(
                "CREATE TEMP VIEW eligible AS SELECT " + ", ".join(
                    [identifier(name) for name in KEYS] + ["filename"]
                    + numeric_projection + category_projection
                ) + " FROM source WHERE proposed_split_eligible IS TRUE "
                "AND candidate_default_12m IN (0, 1) "
                "AND proposed_split IN ('development', 'validation', 'test')"
            )
            leakage = connection.execute("""
                SELECT count(*) FROM source WHERE proposed_split_eligible IS TRUE AND (
                    observation_month IS NULL OR assumed_label_available_month IS NULL
                    OR (proposed_split = 'development' AND
                        (observation_month >= DATE '2018-01-01' OR assumed_label_available_month >= DATE '2018-01-01'))
                    OR (proposed_split = 'validation' AND
                        (observation_month < DATE '2018-01-01' OR observation_month >= DATE '2020-01-01'
                         OR assumed_label_available_month >= DATE '2020-01-01'))
                    OR (proposed_split = 'test' AND
                        (observation_month < DATE '2020-01-01' OR observation_month >= DATE '2024-01-01'))
                )
            """).fetchone()[0]
            if leakage:
                raise ValueError("Prepared eligibility/date rules are inconsistent; review temporal boundaries")
            total = connection.execute("SELECT count(*) FROM source").fetchone()[0]
            eligible_count = connection.execute("SELECT count(*) FROM eligible").fetchone()[0]
            training = connection.execute("SELECT count(*), count(DISTINCT candidate_default_12m) "
                                          "FROM eligible WHERE proposed_split = 'development'").fetchone()
            if training[0] == 0 or training[1] != 2:
                raise ValueError("Development data must contain both observed target classes")
            state["checks"] = {"source_loan_rows": total, "eligible_rows": eligible_count,
                               "excluded_rows": total - eligible_count, "development_rows": training[0],
                               "missing_or_duplicate_ids": bad_ids, "temporal_rule_violations": leakage}
            if "fitted" not in state:
                print("Fitting medians, scales and categories on DEVELOPMENT only ...", flush=True)
                state["fitted"] = fit_preprocessing(connection)
            expressions, feature_names = projection(state["fitted"])
            state["feature_names"] = feature_names
            save(report_root, state)
            for year, source in zip(years, sources):
                target = output / f"model_inputs_{year}.parquet"
                where = "filename = " + literal(source.as_posix())
                counts = dict(connection.execute(
                    "SELECT proposed_split, count(*) FROM eligible WHERE " + where + " GROUP BY proposed_split"
                ).fetchall())
                expected_rows = sum(counts.values())
                expected_names = KEYS + feature_names
                if target.exists():
                    metadata = pq.read_metadata(target)
                    if metadata.num_rows != expected_rows or metadata.schema.names != expected_names:
                        raise ValueError(f"Existing output failed metadata checks: {target}")
                    print(f"Reusing {target.name}", flush=True)
                else:
                    print(f"Preparing {year}: {expected_rows:,} eligible loans ...", flush=True)
                    temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
                    try:
                        connection.execute(
                            "COPY (SELECT " + ", ".join(expressions) + " FROM eligible WHERE " + where
                            + ") TO " + literal(temporary.as_posix())
                            + " (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
                        )
                        metadata = pq.read_metadata(temporary)
                        if metadata.num_rows != expected_rows or metadata.schema.names != expected_names:
                            raise ValueError(f"Output row/schema check failed for {year}")
                        if target.exists():
                            raise FileExistsError(target)
                        temporary.rename(target)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                state["files"][str(year)] = {"file": str(target.relative_to(ROOT)), "rows": expected_rows,
                                            "splits": counts, "bytes": target.stat().st_size}
                save(report_root, state)
        for source, saved_fingerprint in zip(base_sources, fingerprints):
            info = source.stat()
            if (info.st_size, info.st_mtime_ns) != (saved_fingerprint["bytes"], saved_fingerprint["mtime_ns"]):
                raise ValueError(f"Prepared input changed while processing: {source}")
        if performance_fingerprints != [fingerprint(path) for paths in performance_sources.values() for path in paths]:
            raise ValueError("Cleaned performance inputs changed while processing")
        state["status"] = "INPUTS PREPARED - PROVISIONAL RULES REQUIRE REVIEW; NO MODEL FITTED"
        save(report_root, state)
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        save(report_root, state)
        raise
    print(f"Finished. Report: {report_root / 'report.md'}", flush=True)
    print("No model trained. Use only manifest feature_names as future model predictors.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2012, 2024)))
    parser.add_argument("--prepared-run")
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    main(args.years, args.prepared_run, args.memory_limit, args.threads)
