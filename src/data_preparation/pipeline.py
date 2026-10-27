"""Stage 2: prepare Parquet data and write a Markdown change report.

Run: python src/data_preparation/pipeline.py
Default: all full years from 2012 through 2025, reusing completed files.
Examples: --years 2012 2013; --years 2012 2013 2014 2015
Memory: --memory-limit 4GB --threads 2 (DuckDB engine budget, not a hard process cap).
Normal runs automatically reuse a saved run for the selected years.
Adding years reuses the earlier run when its years are a subset of the selection.
Resume: --resume-run <run_id> selects a specific saved run.
Fresh start: --new-run (only when intentionally repeating all work).

Source: data/raw/freddie_mac/parquets/<year>/
New data: data/prepared/<run_id>/<year>/
Report: results/data_preparation/<run_id>/report.md
Original data is never overwritten. No models are fitted. Research labels and
splits are provisional and must be reviewed before downstream model training.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import tempfile
from uuid import uuid4

import pyarrow.parquet as pq

try:
    import duckdb
except ImportError as error:
    raise SystemExit("Install the Stage 2 dependency: python -m pip install duckdb==2.0.0") from error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data_extracting.schema import get_headers

if __package__:
    from .cleaning import clean_file
    from .cohort import prepare_cohort
    from .report import write_report
    from .rules import CREDIT_EXIT_CODES, FEATURES, MISSING_CODES, RANGES
else:
    from cleaning import clean_file
    from cohort import prepare_cohort
    from report import write_report
    from rules import CREDIT_EXIT_CODES, FEATURES, MISSING_CODES, RANGES


def save_progress(report_path, run):
    # Replace the checkpoint atomically so interruption cannot truncate it.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=report_path.parent,
            prefix=".manifest_", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(run, indent=2, default=str))
        temporary.replace(report_path.with_name("manifest.json"))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    write_report(report_path, run)


def find_saved_run(years, lag_months, rules):
    """Select saved progress; never silently repeat overlapping saved work."""
    folder = ROOT / "results" / "data_preparation"
    candidates, blocked = [], []
    requested = set(years)
    canonical_rules = json.loads(json.dumps(rules))
    for manifest in sorted(folder.glob("*/manifest.json"), reverse=True):
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        saved_years = set(saved.get("years", []))
        if not saved_years.intersection(requested):
            continue
        run_id = manifest.parent.name
        compatible = (
            re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{6}", run_id)
            and saved.get("run_id") == run_id
            and saved_years.issubset(requested)
            and saved.get("lag_months") == lag_months
            and saved.get("rules") == canonical_rules
            and saved.get("duckdb_version") == duckdb.__version__
            and (ROOT / "data" / "prepared" / run_id).is_dir()
        )
        if compatible:
            # Prefer the most completed files, then the newest run on ties.
            completed = sum(len(result.get("files", []))
                            for result in saved.get("results", {}).values())
            candidates.append((completed, run_id))
        else:
            blocked.append(run_id)
    if candidates:
        return max(candidates)[1]
    if blocked:
        raise ValueError(
            "Saved work exists but cannot be reused with this year selection/rules: "
            + ", ".join(blocked)
            + ". Use the original selection, or --new-run only to intentionally repeat work."
        )
    return None


def main(years, memory_limit="4GB", threads=2, lag_months=1, resume_run=None, new_run=False):
    years = sorted(set(years))
    if not years or any(year < 2012 or year > 2025 for year in years):
        raise ValueError("Choose full cohort years from 2012 through 2025")
    if threads not in (1, 2) or lag_months < 0:
        raise ValueError("Use one/two threads and a non-negative reporting lag")
    if new_run and resume_run is not None:
        raise ValueError("Choose either --new-run or --resume-run, not both")
    if resume_run is not None and not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{6}", resume_run):
        raise ValueError("Invalid resume run ID")
    run_id = resume_run or (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
    output_root = ROOT / "data" / "prepared" / run_id
    report_path = ROOT / "results" / "data_preparation" / run_id / "report.md"
    run = {"run_id": run_id, "status": "RUNNING", "years": years,
           "lag_months": lag_months, "memory_limit": memory_limit,
           "threads": threads, "duckdb_version": duckdb.__version__,
           "rules": {"missing_codes": MISSING_CODES, "numeric_ranges": RANGES,
                     "features": FEATURES, "credit_exit_codes": CREDIT_EXIT_CODES,
                     "horizon_months": 12, "baseline": "first valid monthly record; current; positive balance; FRM",
                     "early_exit_policy": "censored", "labels_and_splits": "provisional"},
           "sources": [], "results": {}}
    if resume_run is None and not new_run:
        resume_run = find_saved_run(years, lag_months, run["rules"])
        if resume_run is not None:
            run_id = resume_run
            run["run_id"] = run_id
            output_root = ROOT / "data" / "prepared" / run_id
            report_path = ROOT / "results" / "data_preparation" / run_id / "report.md"
            print(f"Automatically reusing saved run: {run_id}", flush=True)
    previous = {}
    if resume_run is not None:
        manifest = report_path.with_name("manifest.json")
        if not manifest.is_file() or not output_root.is_dir():
            raise FileNotFoundError("Resume requires the existing prepared folder and its manifest.json")
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not set(previous.get("years", [])).issubset(years):
            raise ValueError("Keep the saved run's years selected; additional years can be added")
        for key in ("lag_months", "rules", "duckdb_version"):
            current_value = json.loads(json.dumps(run[key]))
            if previous.get(key) != current_value:
                raise ValueError(f"Cannot reuse this run: {key} changed. Start a new run.")
        # Retain all cached summaries even if a resumed process stops before
        # reaching later files/years again.
        run["results"] = json.loads(json.dumps(previous.get("results", {})))
    run["cleaning_mode"] = "single source pass; bounded Arrow batches"
    connection = None
    progress_started = False
    try:
        # Preflight all requested files before producing cleaned datasets.
        sources = {}
        for year in years:
            sources[year] = []
            for quarter in range(1, 5):
                for kind in ("orig", "perf"):
                    path = ROOT / "data" / "raw" / "freddie_mac" / "parquets" / str(year) / f"{kind}_{year}Q{quarter}.parquet"
                    if not path.is_file():
                        raise FileNotFoundError(f"Run Stage 1 first; missing input: {path}")
                    metadata = pq.read_metadata(path)
                    if metadata.schema.names != get_headers(kind) or metadata.num_rows == 0:
                        raise ValueError(f"Unexpected or empty input: {path}")
                    run["sources"].append({"path": str(path.relative_to(ROOT)),
                                           "bytes": path.stat().st_size,
                                           "mtime_ns": path.stat().st_mtime_ns,
                                           "rows": metadata.num_rows})
                    sources[year].append((path, kind))
        if previous:
            old_sources = sorted(previous.get("sources", []), key=lambda row: row["path"])
            saved_years = set(previous["years"])
            new_sources = sorted(
                (row for row in run["sources"]
                 if int(Path(row["path"]).parent.name) in saved_years),
                key=lambda row: row["path"],
            )
            if old_sources != new_sources:
                # Leave the existing report/manifest intact if inputs changed.
                raise ValueError("Resume inputs changed in size, mtime, row count or file list; start a new run")
        output_root.mkdir(parents=True, exist_ok=bool(resume_run))
        spill = output_root / "_spill"
        spill.mkdir(exist_ok=bool(resume_run))
        connection = duckdb.connect(config={
            "memory_limit": memory_limit, "threads": str(threads),
            "temp_directory": spill.as_posix(), "max_temp_directory_size": "20GB",
            "preserve_insertion_order": "false",
        })
        connection.execute("SET enable_progress_bar = true")
        save_progress(report_path, run)
        progress_started = True
        for year in years:
            print(f"\nPreparing {year} ...", flush=True)
            output = output_root / str(year)
            output.mkdir(exist_ok=bool(resume_run))
            cached = previous.get("results", {}).get(str(year), {})
            cached_files = {row["file"]: row for row in cached.get("files", [])}
            result = json.loads(json.dumps(cached)) if cached else {"files": [], "changes": []}
            result.update({"output": str(output.relative_to(ROOT)), "reused_files": 0})
            run["results"][str(year)] = result
            cleaned = {"orig": [], "perf": []}
            for source, kind in sources[year]:
                target = output / source.name
                if target.exists():
                    file_counts = cached_files.get(source.name)
                    changes = [row for row in cached.get("changes", []) if row["file"] == source.name]
                    metadata = pq.read_metadata(target)
                    if (file_counts is None or metadata.num_rows != file_counts["rows_after"]
                            or target.stat().st_size != file_counts["bytes_after"]
                            or metadata.schema.names != get_headers(kind)
                            or len(changes) != len(get_headers(kind))
                            or {row["column"] for row in changes} != set(get_headers(kind))):
                        raise ValueError(f"Cannot reuse uncached/inconsistent output: {target}. Start a new run.")
                    result["reused_files"] += 1
                    print(f"Reusing completed cleaned file: {source.name}", flush=True)
                else:
                    file_counts, changes = clean_file(connection, source, target, get_headers(kind))
                result["files"] = [row for row in result["files"] if row["file"] != source.name] + [file_counts]
                result["changes"] = [row for row in result["changes"] if row["file"] != source.name] + changes
                cleaned[kind].append(target)
                save_progress(report_path, run)
            blank_ids = {
                label: sum(row["null_after"] for row in result["changes"]
                           if row["column"] == "LOAN IDENTIFIER" and row["file"].startswith(prefix))
                for label, prefix in (("origination", "orig_"), ("performance", "perf_"))
            }
            cohort_path = output / "cohort_candidates.parquet"
            if cohort_path.exists():
                if ("cohort" not in cached or pq.read_metadata(cohort_path).num_rows
                        != cached["cohort"]["metrics"]["loan_ids"]):
                    raise ValueError("An incomplete/uncached cohort exists; start a new run rather than overwrite it")
                result["cohort"] = cached["cohort"]
                print(f"Reusing completed cohort for {year}", flush=True)
            else:
                result["cohort"] = prepare_cohort(
                    connection, cleaned["orig"], cleaned["perf"], output, year,
                    lag_months, blank_ids=blank_ids,
                )
            save_progress(report_path, run)
        run["status"] = "PREPARED - RESEARCH RULES REQUIRE REVIEW"
    except Exception as error:
        run["status"] = "FAILED"
        run["error"] = str(error)
        if progress_started or not previous:
            save_progress(report_path, run)
        print(f"Failed. Review the report: {report_path}", flush=True)
        raise
    finally:
        if connection is not None:
            connection.close()
    save_progress(report_path, run)
    print(f"Finished. Report: {report_path}", flush=True)
    print(f"Prepared data: {output_root}", flush=True)
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2012, 2026)))
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--lag-months", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume-run", help="Select a saved run; otherwise selected automatically")
    mode.add_argument("--new-run", action="store_true", help="Intentionally repeat work in a new folder")
    args = parser.parse_args()
    main(args.years, args.memory_limit, args.threads, args.lag_months, args.resume_run, args.new_run)
