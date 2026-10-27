"""Prepare and check the complete 2012-2025 local mortgage dataset.

Run this file in PyCharm, or: python src/data_extracting/pipeline.py
Optional: --workers 1 (default: two simultaneous conversions).
Report: results/data_extracting/pipeline_2012_2025.md

Per required file: keep existing Parquet; otherwise use TXT; otherwise extract
that TXT from a local ZIP, then convert. No automatic downloads or deletions.
Existing Parquets do not require TXT. New conversions are compared with TXT.
Invalid existing Parquets or unreadable archives are reported, never overwritten.
The 2026 cohort is excluded.
"""

import argparse
from multiprocessing import freeze_support
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__:
    from .convert_missing_parquets import main as convert
    from .compare_parquets import audit_pipeline
else:
    from convert_missing_parquets import main as convert
    from compare_parquets import audit_pipeline

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "freddie_mac"
REPORT = PROJECT_ROOT / "results" / "data_extracting" / "pipeline_2012_2025.md"
YEARS = tuple(range(2012, 2026))
EXPECTED_FILES = tuple(
    Path(str(year)) / f"{kind}_{year}Q{quarter}"
    for year in YEARS
    for quarter in range(1, 5)
    for kind in ("orig", "perf")
)


def write_failure(stage, details):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Pipeline 2012-2025\n\nStatus: FAIL\n\n"
        f"Stage: {stage}\n\n{details}\n", encoding="utf-8",
    )
    print(f"FAIL: {stage}. See {REPORT}", flush=True)


def main(workers=2):
    sources = [
        (DATA_ROOT / "extracted" / path.with_suffix(".txt"), path.name.split("_", 1)[0])
        for path in EXPECTED_FILES
    ]
    print(
        f"Checking {len(sources)} required files for 2012-2025. "
        "Priority: Parquet -> TXT -> local ZIP.",
        flush=True,
    )
    try:
        created = convert(workers=workers, sources=sources)
    except Exception as error:
        write_failure("Preparation", str(error))
        raise

    print(
        "Auditing Parquets; comparing only newly converted files with TXT ...",
        flush=True,
    )
    try:
        success = audit_pipeline(sources, created, REPORT)
    except Exception as error:
        write_failure("Audit", str(error))
        raise
    status = "BASIC PASS" if success else "FAIL"
    print(f"Pipeline status: {status}. Report: {REPORT}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    sys.exit(main(workers=parser.parse_args().workers))
