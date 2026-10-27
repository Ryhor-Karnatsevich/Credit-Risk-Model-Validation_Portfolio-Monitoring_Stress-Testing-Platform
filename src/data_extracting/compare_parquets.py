"""Compare TXT and Parquet files and write a basic Markdown report.

Run in PyCharm, or: python src/data_extracting/compare_parquets.py
Optional: --years 2012 2013 2014 2015
Output: results/data_extracting/parquet_comparison.md

Counts TXT rows without loading whole files. Checks Parquet metadata and the
first 100 rows of values. This is not a full value-by-value validation.
"""

import argparse
import csv
from itertools import islice
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__:
    from .schema import get_headers
else:
    from schema import get_headers

ROOT = PROJECT_ROOT / "data" / "raw" / "freddie_mac"
OUTPUT = PROJECT_ROOT / "results" / "data_extracting" / "parquet_comparison.md"
SAMPLE_ROWS = 100


def count_rows(path):
    rows, last = 0, b""
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            rows += block.count(b"\n")
            last = block[-1:]
    return rows + int(bool(last) and last != b"\n")


def check_file(source, parquet, kind):
    text_rows = count_rows(source)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        sample = list(islice(csv.reader(handle, delimiter="|"), SAMPLE_ROWS))
    columns = get_headers(kind)
    text_width_ok = bool(sample) and all(len(row) == len(columns) for row in sample)
    with pq.ParquetFile(parquet) as reader:
        parquet_rows = reader.metadata.num_rows
        schema = pa.schema([(name, pa.string()) for name in columns])
        columns_ok = text_width_ok and reader.schema_arrow.equals(schema)
        batch = next(reader.iter_batches(batch_size=SAMPLE_ROWS), None)
        if batch is None:
            converted = []
        else:
            converted_columns = [column.to_pylist() for column in batch.columns]
            converted = [list(row) for row in zip(*converted_columns)]
        sample_ok = sample == converted
    passed = text_rows == parquet_rows and columns_ok and sample_ok
    return text_rows, parquet_rows, columns_ok, sample_ok, passed


def audit_pipeline(sources, created, output):
    """Check complete expected coverage without requiring TXT for existing outputs."""
    created = set(created)
    lines = [
        "# Data preparation: 2012-2025", "",
        "Existing Parquets: readable footer, expected schema and first 100 rows readable.",
        "New conversions: TXT row counts, schema and first 100 values compared.",
        "Neither check is a full value-by-value validation.", "",
        "| File | Parquet rows | Check | Result |", "|---|---:|---|---|",
    ]
    passed = compared = 0
    for source, kind in sources:
        parquet = ROOT / "parquets" / source.parent.name / (source.stem + ".parquet")
        label = f"{source.parent.name}/{parquet.name}"
        check = "Parquet only; TXT not required"
        try:
            if parquet in created:
                check = "New conversion compared with TXT"
                _, rows, _, _, ok = check_file(source, parquet, kind)
                compared += 1
            else:
                with pq.ParquetFile(parquet) as reader:
                    rows = reader.metadata.num_rows
                    expected = pa.schema([(name, pa.string()) for name in get_headers(kind)])
                    batch = next(reader.iter_batches(batch_size=SAMPLE_ROWS), None)
                    ok = (rows > 0 and reader.schema_arrow.equals(expected)
                          and batch is not None and batch.num_rows == min(rows, SAMPLE_ROWS))
            passed += int(ok)
            lines.append(f"| {label} | {rows:,} | {check} | {'PASS' if ok else 'FAIL'} |")
        except Exception as error:
            message = str(error).replace("\n", " ").replace("|", "/")
            lines.append(f"| {label} | - | {check}: {message} | ERROR |")
    success = passed == len(sources)
    lines += ["", f"Expected: {len(sources)}. Passed: {passed}. "
              f"New TXT comparisons: {compared}. Status: {'BASIC PASS' if success else 'FAIL'}."]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return success


def main(years=None, output=None):
    report_path = Path(output) if output is not None else OUTPUT
    extracted = ROOT / "extracted"
    if not extracted.is_dir():
        raise FileNotFoundError(extracted)
    if years:
        folders = [extracted / str(year) for year in sorted(set(years))]
    else:
        folders = sorted(path for path in extracted.iterdir()
                         if path.is_dir() and re.fullmatch(r"[0-9]{4}", path.name))
    lines = [
        "# TXT and Parquet comparison", "",
        "Checks: row counts, expected column names/order/string types, and the first 100 rows.",
        "TXT row counts use line breaks; these loan files have one record per line.",
        "A basic pass does not prove that all values match or all Parquet pages are readable.", "",
        "| File | TXT rows | Parquet rows | Columns | First 100 rows | Result |",
        "|---|---:|---:|---|---|---|",
    ]
    sizes, problems = [], []
    checked = passed = missing = errors = 0
    for folder in folders:
        if not folder.is_dir():
            problems.append(f"Source year folder missing: {folder.name}")
            errors += 1
            continue
        for source in sorted(folder.glob("*.txt")):
            match = re.fullmatch(rf"(orig|perf)_{folder.name}Q[1-4]\.txt", source.name)
            if not match:
                continue
            label = f"{folder.name}/{source.stem}"
            parquet = ROOT / "parquets" / folder.name / (source.stem + ".parquet")
            if not parquet.is_file():
                missing += 1
                lines.append(f"| {label} | - | - | - | - | Not converted yet |")
                continue
            print(f"Checking {label} ...", flush=True)
            try:
                text_rows, parquet_rows, columns_ok, sample_ok, ok = check_file(
                    source, parquet, match.group(1)
                )
                checked += 1
                passed += int(ok)
                lines.append(
                    f"| {label} | {text_rows:,} | {parquet_rows:,} | "
                    f"{'Match' if columns_ok else 'Mismatch'} | "
                    f"{'Match' if sample_ok else 'Mismatch'} | "
                    f"{'Basic pass' if ok else 'FAIL'} |"
                )
                sizes.append((label, source.stat().st_size, parquet.stat().st_size))
                print(f"  {'Basic pass' if ok else 'FAIL'}", flush=True)
            except Exception as error:
                errors += 1
                lines.append(f"| {label} | - | - | - | - | ERROR |")
                problems.append(f"{label}: {str(error).replace(chr(10), ' ')}")
                print(f"  ERROR: {error}", flush=True)
    lines += [
        "", "## File sizes", "",
        "| File | TXT (MiB) | Parquet (MiB) | Reduction |",
        "|---|---:|---:|---:|",
    ]
    for label, text_size, parquet_size in sizes:
        reduction = f"{100 * (1 - parquet_size / text_size):.2f}%" if text_size else "N/A"
        lines.append(f"| {label} | {text_size / 2**20:,.2f} | {parquet_size / 2**20:,.2f} | {reduction} |")
    lines += [
        "", f"Compared: {checked}. Basic passes: {passed}. "
        f"Failed comparisons: {checked - passed}. Errors: {errors}. Not converted: {missing}.",
    ]
    if problems:
        lines += ["", "## Errors", ""] + [f"- {message}" for message in problems]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved: {report_path}", flush=True)
    return {"checked": checked, "passed": passed, "failed": checked - passed,
            "errors": errors, "missing": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int)
    main(parser.parse_args().years)
