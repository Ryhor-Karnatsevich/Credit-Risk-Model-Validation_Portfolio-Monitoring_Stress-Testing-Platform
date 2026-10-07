"""Print a memory-efficient inventory of a Freddie Mac origination year."""

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "freddie_mac"


def count_rows(path):
    """Count records in binary chunks, including a final unterminated line."""
    rows = 0
    last_byte = b""
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            rows += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return rows + int(bool(last_byte) and last_byte != b"\n")


def show_info(year, count=True):
    folder = SOURCE_DIR / "extracted" / str(year)
    if not folder.is_dir():
        raise FileNotFoundError(f"Year folder does not exist: {folder}")
    files = sorted(folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No text files found in {folder}")

    print(f"Mortgage origination cohort: {year}", flush=True)
    print("orig: one row per mortgage; perf: one row per mortgage per month.")
    print("The cohort year is not the end date of its performance history.\n")
    for kind, header_name in (
        ("orig", "origination_data_file_header.txt"),
        ("perf", "performance_data_file_header.txt"),
    ):
        header_path = SOURCE_DIR / "headers" / header_name
        headers = header_path.read_text(encoding="utf-8-sig").strip().split("|")
        print(f"{kind.upper()} columns ({len(headers)}):")
        for index, name in enumerate(headers, 1):
            print(f"  {index:2}. {name}")
        print()
        total_rows = 0
        selected = [path for path in files if path.name.startswith(f"{kind}_")]
        if not selected:
            raise FileNotFoundError(f"No {kind} files in {folder}")
        for path in selected:
            with path.open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.reader(file, delimiter="|"), None)
            if row is None or len(row) != len(headers):
                raise ValueError(f"Header/first-row column mismatch: {path.name}")
            size = path.stat().st_size / 1024**3
            print(f"  {path.name}: {size:.3f} GiB", flush=True)
            if count:
                rows = count_rows(path)
                total_rows += rows
                print(f"    Records: {rows:,}", flush=True)
            preview = dict(zip(headers, row))
            keys = (
                ["LOAN IDENTIFIER", "CLASSIC FICO", "ORIGINAL UPB", "ORIGINAL LOAN-TO-VALUE (LTV)"]
                if kind == "orig"
                else ["LOAN IDENTIFIER", "PERIOD", "CURRENT ACTUAL UPB", "CURRENT LOAN DELINQUENCY STATUS"]
            )
            print("    First record: " + "; ".join(f"{key}={preview[key]}" for key in keys))
        if count:
            print(f"  Total {kind} shape: {total_rows:,} rows x {len(headers)} columns")
        print()

    print(f"Total raw size: {sum(path.stat().st_size for path in files) / 1024**3:.3f} GiB")
    print("This is an inventory, not a data-quality or default-rate assessment.")
    if not count:
        print("Row counting skipped; remove --skip-row-count to show dataset shape.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--count-rows", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-row-count", action="store_true", help="Skip the full-file scan for a quick column preview")
    args = parser.parse_args()
    show_info(args.year, count=not args.skip_row_count)
