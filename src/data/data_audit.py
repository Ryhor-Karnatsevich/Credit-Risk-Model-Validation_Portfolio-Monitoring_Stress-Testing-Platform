"""Audit one annual Freddie Mac dataset in chunks; raw files are never changed.

Run: python src/data/data_audit.py --year 2017
Outputs: results/audit_2017/
This script does not construct the model target or certify modelling readiness.
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "freddie_mac"
MONTH_BASE = 1990 * 12

# Field-specific codes: a real zero is never counted as missing.
# This is a deliberately limited register; review other categorical codes separately.
MISSING_CODES = {
    "CLASSIC FICO": {"9999"},
    "VANTAGESCORE 4.0": {"9999"},
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": {"999"},
    "NUMBER OF UNITS": {"99"},
    "OCCUPANCY STATUS": {"9"},
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": {"999"},
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": {"999"},
    "ORIGINAL LOAN-TO-VALUE (LTV)": {"999"},
    "NUMBER OF BORROWERS": {"99"},
    "FIRST TIME HOMEBUYER INDICATOR": {"9"},
    "CHANNEL": {"9"},
    "PROPERTY TYPE": {"99"},
    "LOAN PURPOSE": {"9"},
    "ESTIMATED LOAN-TO-VALUE (ELTV)": {"999"},
}
# Conservative checks for fields with clear numeric interpretation.
NUMERIC_RANGES = {
    "CLASSIC FICO": (300, 850),
    "VANTAGESCORE 4.0": (300, 850),
    "NUMBER OF UNITS": (1, 4),
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": (0, 55),
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": (0, 65),
    "ORIGINAL UPB": (0, None),
    "ORIGINAL INTEREST RATE": (0, None),
    "ORIGINAL LOAN TERM": (1, None),
    "CURRENT ACTUAL UPB": (0, None),
    "CURRENT INTEREST RATE": (0, None),
    "CURRENT NON-INTEREST BEARING UPB": (0, None),
    "LOAN AGE": (0, None),
}
DATE_FIELDS = {
    "FIRST PAYMENT DATE", "MATURITY DATE", "PERIOD", "ZERO BALANCE EFFECTIVE DATE"
}


@dataclass(slots=True)
class History:
    # One bit per calendar month avoids retaining millions of raw monthly rows.
    months: int = 0
    records: int = 0
    valid_records: int = 0
    duplicates: int = 0
    termination_seen: bool = False
    numeric_3plus_seen: bool = False
    credit_exit_seen: bool = False


def month_numbers(values):
    """Validate YYYYMM and map to month offsets; allowed years are 1990-2100."""
    shape_ok = values.str.fullmatch(r"[0-9]{6}", na=False)
    numeric = pd.to_numeric(values, errors="coerce")
    years = numeric // 100
    months = numeric % 100
    valid = (shape_ok & years.between(1990, 2100) & months.between(1, 12)).fillna(False)
    offsets = (years * 12 + months - 1 - MONTH_BASE).where(valid)
    return valid, offsets


def month_label(offset):
    year, month = divmod(offset + MONTH_BASE, 12)
    return f"{year:04d}-{month + 1:02d}"


def headers_for(kind):
    filename = ("origination" if kind == "orig" else "performance") + "_data_file_header.txt"
    headers = (SOURCE_DIR / "headers" / filename).read_text(
        encoding="utf-8-sig"
    ).strip().split("|")
    if len(headers) != len(set(headers)):
        raise ValueError(f"Duplicate column names in {filename}")
    return headers


def audit_columns(frame, totals):
    parsed_months = {}
    for column in frame:
        values = frame[column]
        stats = totals[column]
        blank = values.eq("")
        sentinel = values.isin(MISSING_CODES.get(column, set()))
        stats["rows"] += len(frame)
        stats["blank"] += int(blank.sum())
        stats["sentinel"] += int(sentinel.sum())
        present = ~(blank | sentinel)
        numeric = None
        if column in NUMERIC_RANGES:
            numeric = pd.to_numeric(values.where(present), errors="coerce")
            stats["invalid_numeric"] += int((present & numeric.isna()).sum())
            low, high = NUMERIC_RANGES[column]
            outside = numeric.lt(low)
            if high is not None:
                outside |= numeric.gt(high)
            stats["out_of_range"] += int((present & outside).sum())
        if column in DATE_FIELDS:
            valid, offsets = month_numbers(values)
            parsed_months[column] = (valid, offsets)
            stats["invalid_month"] += int((present & ~valid).sum())
    return parsed_months


def save_columns(totals, path):
    rows = []
    for column, counts in totals.items():
        total = counts["rows"]
        blank = counts["blank"]
        sentinel = counts["sentinel"]
        rows.append({
            "column": column,
            "rows": total,
            "blank_values": blank,
            "known_missing_codes": sentinel,
            "blank_or_known_missing": blank + sentinel,
            "missing_percent": round(100 * (blank + sentinel) / total, 4) if total else None,
            "invalid_numeric_values": counts["invalid_numeric"],
            "out_of_range_values": counts["out_of_range"],
            "invalid_YYYYMM_values": counts["invalid_month"],
            "missing_codes_checked": ", ".join(sorted(MISSING_CODES.get(column, set()))),
        })
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return table


def stream_file(path, headers, chunk_size):
    # Read without names so unexpected file widths cannot be silently truncated.
    for frame in pd.read_csv(
        path, sep="|", header=None, dtype="string[pyarrow]",
        keep_default_na=False, encoding="utf-8-sig",
        chunksize=chunk_size, on_bad_lines="error",
    ):
        if frame.shape[1] != len(headers):
            raise ValueError(
                f"Schema mismatch in {path.name}: expected {len(headers)} "
                f"columns, found {frame.shape[1]}"
            )
        frame.columns = headers
        # Normalise whitespace for analysis only.
        for column in frame:
            frame[column] = frame[column].fillna("").str.strip()
        yield frame


def grouped_histories(frame, parsed, serious, credit_exit):
    """Reduce monthly rows to one summary per loan using NumPy operations."""
    ids = frame["LOAN IDENTIFIER"]
    selected = ids.ne("").to_numpy(dtype=bool)
    if not selected.any():
        return {}
    codes, loan_ids = pd.factorize(ids[selected], sort=False)
    # Avoid sorting when the file already groups each loan's records together.
    order = (
        np.arange(len(codes))
        if np.all(codes[1:] >= codes[:-1])
        else np.argsort(codes, kind="stable")
    )
    ordered_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    period_valid, offsets = parsed["PERIOD"]
    valid = period_valid.to_numpy(dtype=bool)[selected][order]
    months = offsets.fillna(0).to_numpy(dtype=np.int64)[selected][order]
    valid_counts = np.add.reduceat(valid.astype(np.int64), starts)

    exits = frame["ZERO BALANCE CODE"]
    flags = [
        exits.ne("").to_numpy(dtype=bool),
        serious.to_numpy(dtype=bool),
        credit_exit.to_numpy(dtype=bool),
    ]
    reduced_flags = [np.logical_or.reduceat(flag[selected][order], starts) for flag in flags]

    # Encode months in uint64 words. OR reduction gives exact unique months,
    # including unsorted rows and repeated loan-months across chunk boundaries.
    words = months // 64
    bits = np.left_shift(np.uint64(1), (months % 64).astype(np.uint64))
    reduced_words = []
    for word in np.unique(words[valid]):
        mask = np.where(valid & (words == word), bits, np.uint64(0))
        reduced_words.append((int(word) * 64, np.bitwise_or.reduceat(mask, starts)))

    result = {}
    for index, code in enumerate(ordered_codes[starts]):
        month_mask = 0
        for shift, values in reduced_words:
            month_mask |= int(values[index]) << shift
        result[str(loan_ids[code])] = History(
            months=month_mask,
            records=int(counts[index]),
            valid_records=int(valid_counts[index]),
            termination_seen=bool(reduced_flags[0][index]),
            numeric_3plus_seen=bool(reduced_flags[1][index]),
            credit_exit_seen=bool(reduced_flags[2][index]),
        )
    return result


def merge_histories(destination, source):
    """Merge summaries exactly, even when a loan occurs in different files."""
    for loan_id, incoming in source.items():
        current = destination.get(loan_id)
        if current is None:
            destination[loan_id] = incoming
        else:
            current.months |= incoming.months
            current.records += incoming.records
            current.valid_records += incoming.valid_records
            current.termination_seen |= incoming.termination_seen
            current.numeric_3plus_seen |= incoming.numeric_3plus_seen
            current.credit_exit_seen |= incoming.credit_exit_seen


def audit_file(kind, path, headers, chunk_size):
    """Worker: read one quarterly file once and return compact loan summaries."""
    started = time.perf_counter()
    last_progress = started
    totals = {column: Counter() for column in headers}
    counters = Counter()
    orig_ids = set()
    histories = {}
    delinquency_codes = Counter()
    termination_codes = Counter()
    rows_read = 0
    print(f"Auditing {path.name} ...", flush=True)
    for frame in stream_file(path, headers, chunk_size):
        rows_read += len(frame)
        counters[f"{kind}_rows"] += len(frame)
        parsed = audit_columns(frame, totals)
        ids = frame["LOAN IDENTIFIER"]
        nonempty_ids = ids.ne("")
        counters[f"{kind}_rows_without_loan_id"] += int((~nonempty_ids).sum())
        if kind == "orig":
            valid_ids = ids[nonempty_ids]
            unique_ids = set(valid_ids)
            counters["duplicate_origination_rows"] += (
                len(valid_ids) - len(unique_ids) + len(unique_ids & orig_ids)
            )
            orig_ids.update(unique_ids)
        else:
            period_valid, _ = parsed["PERIOD"]
            counters["perf_invalid_or_missing_period_rows"] += int((~period_valid).sum())
            statuses = frame["CURRENT LOAN DELINQUENCY STATUS"]
            exits = frame["ZERO BALANCE CODE"]
            delinquency_codes.update(statuses.replace("", "(blank)").value_counts().to_dict())
            termination_codes.update(exits.replace("", "(blank)").value_counts().to_dict())
            status = pd.to_numeric(statuses, errors="coerce")
            serious = (status.ge(3) & status.lt(999)).fillna(False)
            credit_exit = exits.isin({"02", "03", "09"})
            counters["rows_with_numeric_delinquency_3plus"] += int(serious.sum())
            counters["rows_with_credit_exit_code_02_03_09"] += int(credit_exit.sum())
            merge_histories(histories, grouped_histories(frame, parsed, serious, credit_exit))
        now = time.perf_counter()
        if now - last_progress >= 15:
            print(f"  {path.name}: {rows_read:,} rows | {(now - started) / 60:.1f} min", flush=True)
            last_progress = now
    elapsed = time.perf_counter() - started
    print(f"Finished {path.name}: {rows_read:,} rows in {elapsed / 60:.1f} min", flush=True)
    return {
        "kind": kind, "totals": totals, "counters": counters,
        "orig_ids": orig_ids, "histories": histories,
        "delinquency_codes": delinquency_codes, "termination_codes": termination_codes,
        "file": {
            "file": path.name, "kind": kind, "rows": rows_read,
            "bytes": path.stat().st_size, "elapsed_seconds": round(elapsed, 2),
        },
    }

def audit_year(year=2017, chunk_size=250_000, workers=2):
    folder = SOURCE_DIR / "extracted" / str(year)
    files_by_kind = {
        kind: [folder / f"{kind}_{year}Q{quarter}.txt" for quarter in range(1, 5)]
        for kind in ("orig", "perf")
    }
    missing_files = [str(path) for files in files_by_kind.values() for path in files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing quarterly files:\n" + "\n".join(missing_files))

    output = PROJECT_ROOT / "results" / f"audit_{year}"
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "year": year,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "workers": workers,
        "chunk_size": chunk_size,
        "scope": "One origination cohort year, including all available subsequent performance months",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    orig_ids = set()
    histories = {}
    file_rows = []
    column_totals = {}
    counters = Counter()
    delinquency_codes = Counter()
    termination_codes = Counter()

    headers_by_kind = {kind: headers_for(kind) for kind in files_by_kind}
    column_totals = {
        kind: {column: Counter() for column in headers}
        for kind, headers in headers_by_kind.items()
    }
    started = time.perf_counter()
    print(f"Workers: {workers} | Chunk size: {chunk_size:,} | Grouped monthly calculations", flush=True)
    # Consume completed futures promptly; do not keep four large result sets alive.
    jobs = [(kind, path) for kind, files in files_by_kind.items() for path in files]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        pending = {}
        iterator = iter(jobs)

        def submit_next():
            job = next(iterator, None)
            if job is not None:
                kind, path = job
                future = pool.submit(audit_file, kind, path, headers_by_kind[kind], chunk_size)
                pending[future] = path.name

        for _ in range(workers):
            submit_next()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                result = future.result()
                kind = result["kind"]
                for column, stats in result["totals"].items():
                    column_totals[kind][column].update(stats)
                counters.update(result["counters"])
                # Detect duplicate original IDs across quarterly files too.
                counters["duplicate_origination_rows"] += len(orig_ids & result["orig_ids"])
                orig_ids.update(result["orig_ids"])
                merge_histories(histories, result["histories"])
                delinquency_codes.update(result["delinquency_codes"])
                termination_codes.update(result["termination_codes"])
                file_rows.append(result["file"])
                del result
                print(f"Completed {len(file_rows)}/8 files", flush=True)
                submit_next()
            # Future objects hold their return values until released.
            completed.clear()
            del future
    file_rows.sort(key=lambda item: (item["kind"], item["file"]))
    summary["scan_elapsed_seconds"] = round(time.perf_counter() - started, 2)

    orig_quality = save_columns(column_totals["orig"], output / "orig_column_quality.csv")
    perf_quality = save_columns(column_totals["perf"], output / "perf_column_quality.csv")
    pd.DataFrame(file_rows).to_csv(output / "file_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sorted(delinquency_codes.items()), columns=["delinquency_code", "rows"]).to_csv(
        output / "delinquency_codes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(sorted(termination_codes.items()), columns=["termination_code", "rows"]).to_csv(
        output / "termination_codes.csv", index=False, encoding="utf-8-sig"
    )

    last_global = max((history.months.bit_length() - 1 for history in histories.values() if history.months), default=None)
    coverage_path = output / "loan_monthly_coverage.csv"
    with coverage_path.open("w", encoding="utf-8-sig", newline="") as file:
        import csv
        writer = csv.writer(file)
        writer.writerow([
            "loan_id", "in_orig", "performance_rows", "unique_reported_months",
            "first_month", "last_month", "missing_months_between_first_and_last",
            "duplicate_loan_month_rows", "next_12_months_continuously_reported",
            "termination_code_seen", "ends_before_cohort_latest_month_without_termination_code",
            "numeric_3plus_delinquency_seen", "credit_exit_02_03_09_seen",
        ])
        for loan_id, history in histories.items():
            mask = history.months
            present = mask.bit_count()
            history.duplicates = history.valid_records - present
            first = (mask & -mask).bit_length() - 1 if mask else None
            last = mask.bit_length() - 1 if mask else None
            gaps = last - first + 1 - present if mask else 0
            next_12_complete = bool(mask and ((mask >> (first + 1)) & 4095) == 4095)
            early_end = bool(mask and last < last_global and not history.termination_seen)
            unmatched = loan_id not in orig_ids
            counters["performance_loans_without_origination"] += int(unmatched)
            counters["loans_with_monthly_gaps"] += int(gaps > 0)
            counters["missing_internal_loan_months"] += gaps
            counters["duplicate_loan_month_rows"] += history.duplicates
            counters["loans_with_continuous_next_12_months"] += int(next_12_complete)
            counters["loans_ending_early_without_termination_code"] += int(early_end)
            counters["loans_with_numeric_3plus_delinquency"] += int(history.numeric_3plus_seen)
            counters["loans_with_credit_exit_02_03_09"] += int(history.credit_exit_seen)
            counters["performance_loans_without_valid_months"] += int(not mask)
            writer.writerow([
                loan_id, not unmatched, history.records, present,
                month_label(first) if mask else "", month_label(last) if mask else "",
                gaps, history.duplicates, next_12_complete, history.termination_seen,
                early_end, history.numeric_3plus_seen, history.credit_exit_seen,
            ])
        # Loans with no performance rows must be visible in the coverage report too.
        missing_performance = 0
        for loan_id in orig_ids:
            if loan_id not in histories:
                missing_performance += 1
                writer.writerow([loan_id, True, 0, 0, "", "", "", 0, False, False, False, False, False])

    summary.update(dict(counters))
    summary.update({
        "unique_origination_loans": len(orig_ids),
        "unique_performance_loans": len(histories),
        "origination_loans_without_performance": missing_performance,
        "latest_observed_month": month_label(last_global) if last_global is not None else None,
        "orig_blank_or_known_missing_cells": int(orig_quality["blank_or_known_missing"].sum()),
        "perf_blank_or_known_missing_cells": int(perf_quality["blank_or_known_missing"].sum()),
        "status": "AUDIT_COMPLETED_REVIEW_REQUIRED",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Blank fields can mean not applicable; blank recovery fields do not automatically imply missing usable loss data.",
            "Missing-code checks cover the explicit field-specific register only; other categorical codes need dictionary review.",
            "Range checks are conservative and do not validate every business rule or disclosure format change.",
            "Gaps count absent months inside the observed first-to-last interval, not months before acquisition or after termination.",
            "Early end uses the latest observed month in this cohort, not a verified release cutoff; investigate flags before exclusions.",
            "Continuous next-12-month reporting excludes the first snapshot; it is not a mature-outcome eligibility decision.",
            "Valid terminal outcomes can end a history before 12 months; no automatic non-default or default labels are assigned.",
            "Numeric 3+ delinquency and termination codes are descriptive counts, not the frozen model target.",
            "Latest-release records may contain corrections; point-in-time availability and leakage are not certified.",
        ],
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        f"# Annual data-quality audit: {year}",
        "",
        f"- Origination rows: {counters['orig_rows']:,}",
        f"- Unique origination loans: {len(orig_ids):,}",
        f"- Monthly performance rows: {counters['perf_rows']:,}",
        f"- Unique performance loans: {len(histories):,}",
        f"- Origination loans without performance: {missing_performance:,}",
        f"- Performance loans without origination: {counters['performance_loans_without_origination']:,}",
        f"- Duplicate origination rows: {counters['duplicate_origination_rows']:,}",
        f"- Duplicate loan-month rows: {counters['duplicate_loan_month_rows']:,}",
        f"- Loans with internal reporting gaps: {counters['loans_with_monthly_gaps']:,}",
        f"- Missing internal loan-months: {counters['missing_internal_loan_months']:,}",
        f"- Loans with all 12 following months reported: {counters['loans_with_continuous_next_12_months']:,}",
        "",
        "## Interpretation",
        "",
        *[f"- {note}" for note in summary["limitations"]],
        "",
        "Status: completed checks, review required. No modelling-readiness certificate is issued.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report[:15]))
    print(f"\nReports saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if not 1999 <= args.year <= 2100:
        parser.error("--year must be between 1999 and 2100")
    try:
        audit_year(args.year, args.chunk_size, args.workers)
    except Exception as error:
        summary_file = PROJECT_ROOT / "results" / f"audit_{args.year}" / "summary.json"
        if summary_file.is_file():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            summary.update(status="FAILED", error=str(error))
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise
