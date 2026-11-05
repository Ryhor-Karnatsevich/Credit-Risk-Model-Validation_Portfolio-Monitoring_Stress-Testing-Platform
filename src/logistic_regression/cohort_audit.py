"""Report yearly loan inclusion/exclusion for the saved logistic baseline.

Run this file normally in PyCharm; no parameters are needed. Reads only a few
columns from the corrected, one-row-per-loan candidate tables, never monthly
performance files. Does not clean data, fit/score a model or evaluate test risk.

Output: results/logistic_regression/cohort_audit.md
Optional: --prepared-run ID --threads 1 --memory-limit 2GB
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
POLICY = "verified_repayment_v2"
SPLITS = ("development", "validation", "test")
REQUIRED = {
    "origination_cohort", "observation_month", "candidate_default_12m",
    "outcome_reason", "proposed_split", "proposed_split_eligible",
    "assumed_label_available_month", "target_policy",
}
REASONS = {
    "missing_origination": "No matching origination record",
    "duplicate_origination_id": "More than one origination record for the loan",
    "not_confirmed_fixed_rate": "Loan not confirmed as fixed-rate",
    "missing_performance": "No monthly performance history",
    "invalid_reporting_month": "Invalid reporting date in history",
    "duplicate_loan_month_in_window": "Duplicate monthly records in the outcome window",
    "baseline_not_current_or_already_exited": "Not current, positive-balance and event-free at the starting date",
    "early_exit_censored": "Early exit without verified event-free full repayment",
    "insufficient_followup": "History ends before the 12-month outcome is known",
    "missing_window_months": "Missing monthly records in the outcome window",
    "unknown_status_in_window": "Unknown delinquency status in the outcome window",
    "outside_selected_period": "Observation outside development/validation/test periods",
    "missing_observation_date": "No valid starting observation date",
    "label_not_available": "Outcome not available before the next period starts",
    "split_not_eligible": "Other saved split-eligibility restriction; review required",
    "unknown_outcome": "Unknown outcome; no saved explanation",
}


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def local_path(value):
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def select_baseline(requested=None):
    folder = ROOT / "results" / "logistic_regression"
    if requested is not None:
        if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{6}", requested):
            raise ValueError("Invalid preparation run ID")
        paths = [folder / requested / POLICY / "baseline" / "metrics.json"]
    else:
        paths = folder.glob(f"*/{POLICY}/baseline/metrics.json")
    completed = []
    for path in paths:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status", "").startswith("BASELINE FITTED") and state.get("completed_at_utc"):
            completed.append((datetime.fromisoformat(state["completed_at_utc"]), path, state))
    if not completed:
        raise ValueError("No completed baseline found. This audit describes an existing baseline only.")
    _, metrics_path, baseline = max(completed, key=lambda item: item[0])
    manifest_path = metrics_path.parent.parent / "manifest.json"
    raw = manifest_path.read_bytes()
    inputs = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != baseline["signature"]["input_manifest_sha256"]:
        raise ValueError("Preprocessing changed since baseline fitting; cannot audit it as the same sample.")
    if (not inputs.get("status", "").startswith("INPUTS PREPARED")
            or inputs["signature"]["target_policy"] != POLICY
            or inputs["prepared_run"] != baseline["prepared_run"]):
        raise ValueError("Baseline and corrected preprocessing do not match.")
    if any(inputs["checks"].get(name) != 0 for name in
           ("missing_or_duplicate_ids", "temporal_rule_violations")):
        raise ValueError("Saved ID/date checks did not pass. Review preprocessing first.")
    years = set(map(str, inputs["years"]))
    if set(inputs["files"]) != years or set(inputs["label_updates"]) != years:
        raise ValueError("Missing yearly preprocessing records.")
    return inputs, baseline, metrics_path


def count_year(connection, path, year):
    # Same inclusion predicate as preprocessing.py; one mutually exclusive
    # explanation per excluded row. Underlying problems may overlap.
    query = f"""
        WITH source AS (
            SELECT *, coalesce(proposed_split_eligible IS TRUE
                AND candidate_default_12m IN (0, 1)
                AND proposed_split IN ('development', 'validation', 'test'), FALSE) AS included
            FROM read_parquet({literal(path.as_posix())})
        ), classified AS (
            SELECT *, CASE
                WHEN included THEN 'included'
                WHEN candidate_default_12m IS NULL
                    THEN coalesce(nullif(outcome_reason, ''), 'unknown_outcome')
                WHEN observation_month IS NULL THEN 'missing_observation_date'
                WHEN proposed_split IS NULL OR proposed_split NOT IN ('development', 'validation', 'test')
                    THEN 'outside_selected_period'
                WHEN proposed_split = 'development' AND
                    (assumed_label_available_month IS NULL OR assumed_label_available_month >= DATE '2018-01-01')
                    THEN 'label_not_available'
                WHEN proposed_split = 'validation' AND
                    (assumed_label_available_month IS NULL OR assumed_label_available_month >= DATE '2020-01-01')
                    THEN 'label_not_available'
                ELSE 'split_not_eligible' END AS audit_reason
            FROM source
        )
        SELECT proposed_split, year(observation_month) AS observation_year,
            audit_reason, count(*) AS loans,
            count(*) FILTER (WHERE candidate_default_12m IS NOT NULL
                AND candidate_default_12m NOT IN (0, 1)) AS invalid_targets,
            count(*) FILTER (WHERE origination_cohort IS NULL OR origination_cohort != {year}
                OR target_policy IS NULL OR target_policy != {literal(POLICY)}) AS invalid_source
        FROM classified GROUP BY proposed_split, observation_year, audit_reason
    """
    result = connection.execute(query)
    names = [item[0] for item in result.description]
    groups = [dict(zip(names, row)) for row in result.fetchall()]
    if any(row["invalid_targets"] or row["invalid_source"] for row in groups):
        raise ValueError(f"Unexpected target, vintage or policy in {path}")
    return groups


def add_table(lines, headings, rows):
    lines += ["| " + " | ".join(headings) + " |", "|" + "|".join("---" for _ in headings) + "|"]
    lines += ["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |"
              for row in rows]
    lines.append("")


def main(prepared_run=None, memory_limit="2GB", threads=2):
    inputs, baseline, metrics_path = select_baseline(prepared_run)
    yearly, splits, reasons = [], [], []
    by_observation = defaultdict(Counter)
    total_reasons = Counter()
    total = Counter()
    with duckdb.connect(config={"memory_limit": memory_limit, "threads": str(threads)}) as connection:
        connection.execute("SET enable_progress_bar = false")
        for year in sorted(inputs["years"]):
            print(f"Counting included/excluded loans: {year} ...", flush=True)
            update = inputs["label_updates"][str(year)]
            path = local_path(update["file"])
            expected_path = ROOT / "data" / "prepared" / inputs["prepared_run"] / "target_revisions" / POLICY / str(year) / "cohort_candidates.parquet"
            if path != expected_path.resolve():
                raise ValueError(f"Unexpected corrected cohort path: {path}")
            metadata = pq.read_metadata(path)
            if (not REQUIRED.issubset(metadata.schema.names)
                    or metadata.num_rows != update["stats"]["rows"]
                    or path.stat().st_size != update["bytes"]):
                raise ValueError(f"Corrected cohort differs from its saved report: {path}")
            source_before = (path.stat().st_size, path.stat().st_mtime_ns)
            groups = count_year(connection, path, year)
            if source_before != (path.stat().st_size, path.stat().st_mtime_ns):
                raise ValueError(f"Source changed during audit: {path}")
            available = sum(row["loans"] for row in groups)
            included = Counter()
            excluded = Counter()
            for row in groups:
                key = "included" if row["audit_reason"] == "included" else "excluded"
                observation = row["observation_year"]
                by_observation[observation][key] += row["loans"]
                if key == "included":
                    included[row["proposed_split"]] += row["loans"]
                else:
                    excluded[row["audit_reason"]] += row["loans"]
            saved = inputs["files"][str(year)]
            if (available != metadata.num_rows or sum(included.values()) != saved["rows"]
                    or any(included[split] != saved["splits"].get(split, 0) for split in SPLITS)
                    or available != sum(included.values()) + sum(excluded.values())):
                raise ValueError(f"Year {year}: counts do not reconcile with completed preprocessing.")
            rejected = sum(excluded.values())
            yearly.append([year, f"{available:,}", f"{sum(included.values()):,}", f"{rejected:,}",
                           f"{100 * rejected / available:.2f}%" if available else "-"])
            splits.append([year] + [f"{included[split]:,}" for split in SPLITS])
            reasons.extend([year, REASONS.get(reason, reason), f"{count:,}"]
                           for reason, count in sorted(excluded.items()))
            total_reasons.update(excluded)
            total.update(available=available, included=sum(included.values()), excluded=rejected)
            total.update(included)

    expected = inputs["checks"]
    if (total["available"] != expected["source_loan_rows"]
            or total["included"] != expected["eligible_rows"]
            or total["excluded"] != expected["excluded_rows"]
            or total["development"] != baseline["training"]["rows"]
            or total["validation"] != baseline["validation"]["metrics"]["loans"]):
        raise ValueError("Overall counts differ from saved preprocessing/baseline totals.")

    lines = ["# Loan inclusion and exclusions", "",
             f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"Preparation run: {inputs['prepared_run']}; target policy: {POLICY}.", "",
             "Available means candidate loan IDs in the saved corrected tables, not monthly records or all US mortgages.",
             "Included means retained in model inputs; only development loans trained the model. Test loans remain reserved.", "",
             "## By origination year", "",
             "Year means when the mortgage originated. Model periods use the first observed performance date, which can be later.", ""]
    yearly.append(["Total", f"{total['available']:,}", f"{total['included']:,}", f"{total['excluded']:,}",
                   f"{100 * total['excluded'] / total['available']:.2f}%" if total['available'] else "-"])
    add_table(lines, ["Year", "Available loans", "Included in inputs", "Excluded", "Excluded %"], yearly)
    lines += ["## Where included loans go", ""]
    splits.append(["Total"] + [f"{total[split]:,}" for split in SPLITS])
    add_table(lines, ["Origination year", "Development (training)", "Validation", "Test (reserved)"], splits)
    lines += ["## Exclusion reasons", "",
              "Each excluded loan has one explanation, using saved outcome reasons first, then period/eligibility rules. Counts add up; they do not count every overlapping issue.", ""]
    add_table(lines, ["Reason", "Excluded loans"],
              [[REASONS.get(reason, reason), f"{count:,}"] for reason, count in total_reasons.most_common()])
    lines += ["### Reasons by origination year", ""]
    add_table(lines, ["Year", "Reason", "Excluded loans"], reasons)
    lines += ["## By starting observation year", "",
              "This is the calendar year the loan first appears in performance data, not its origination year. It helps explain differences between vintage totals and model-period totals.", ""]
    add_table(lines, ["Observation year", "Available loans", "Included in inputs", "Excluded"],
              [[year if year is not None else "Unknown", f"{sum(counts.values()):,}",
                f"{counts['included']:,}", f"{counts['excluded']:,}"]
               for year, counts in sorted(by_observation.items(), key=lambda item: (item[0] is None, item[0] or 0))])
    lines += ["## How to read this", "",
              "- A time-boundary exclusion is not necessarily a data defect: the outcome was not yet available when that historical model would have been fitted.",
              "- Verified event-free early repayments are included as 0; unresolved exits are not assumed to be successful repayments.",
              "- Missing predictor values alone do not exclude a loan; preprocessing handles them using development-only settings.",
              "- Different yearly loan counts do not alone indicate a problem. Compare exclusion percentages and reasons.",
              "- This report describes selection. It does not prove excluded and included loans have the same risk, or that probabilities are accurate across years.",
              "- No cleaning, target changes, model fitting, scoring or final-test performance evaluation is performed.", "",
              "## Reconciliation", "",
              "PASS: candidate totals, included rows per year/split and total exclusions agree with saved preprocessing. Development and validation totals agree with the fitted baseline.",
              "Checks use metadata and aggregated counts; individual model-input membership and every source value are not revalidated.", "",
              f"Baseline source: `{metrics_path.relative_to(ROOT).as_posix()}`.", ""]
    destination = ROOT / "results" / "logistic_regression" / "cohort_audit.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Finished. Report: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    main(args.prepared_run, args.memory_limit, args.threads)
