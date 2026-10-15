"""Convert missing Parquet files for every locally extracted year.

Run: python src/data/convert_missing_parquets.py
Input:  data/raw/freddie_mac/extracted/<year>/*.txt
Output: data/raw/freddie_mac/parquets/<year>/*.parquet

Existing outputs are never overwritten. Checking their footer/schema does not
prove that every source row was preserved or that the source has not changed.
Only available quarters are processed, including partial years such as 2026.
Two files are converted simultaneously by default. Use --workers 1 for sequential
conversion. Do not run other conversion scripts against the same outputs.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import freeze_support, get_context
from pathlib import Path
import re
import tempfile
import time

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

if __package__:
    from .schema import get_headers
else:
    from schema import get_headers


ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "freddie_mac"
EXTRACTED = ROOT / "extracted"
PARQUETS = ROOT / "parquets"
BLOCK_SIZE = 16 * 1024 * 1024


def convert_file(source, destination, kind):
    columns = get_headers(kind)
    schema = pa.schema([(name, pa.string()) for name in columns])
    started = last_progress = time.monotonic()
    rows = 0
    temporary = None
    print(f"Converting {source.name} ...", flush=True)
    try:
        # A unique temporary file prevents incomplete output being mistaken for
        # a completed Parquet file. Only this run's temporary file is cleaned up.
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".partial", delete=False,
        ) as handle:
            temporary = Path(handle.name)

        with csv.open_csv(
            source,
            read_options=csv.ReadOptions(
                column_names=columns, block_size=BLOCK_SIZE, encoding="utf8",
            ),
            parse_options=csv.ParseOptions(delimiter="|", ignore_empty_lines=False),
            convert_options=csv.ConvertOptions(
                column_types=schema, strings_can_be_null=False,
                quoted_strings_can_be_null=False, null_values=[],
            ),
        ) as reader, pq.ParquetWriter(
            temporary, schema, compression="zstd", compression_level=3,
            use_dictionary=True,
        ) as writer:
            for batch in reader:
                # Arrow's parser rejects rows with the wrong number of fields.
                writer.write_batch(batch, row_group_size=128_000)
                rows += batch.num_rows
                now = time.monotonic()
                if now - last_progress >= 5:
                    print(f"  {rows:,} rows converted", flush=True)
                    last_progress = now

        metadata = pq.read_metadata(temporary)
        if metadata.num_rows != rows or metadata.num_columns != len(columns):
            raise ValueError(f"Output row/column count mismatch: {source.name}")
        if not metadata.schema.to_arrow_schema().equals(schema):
            raise ValueError(f"Output schema mismatch: {source.name}")
        if destination.exists():
            raise FileExistsError(f"Will not overwrite: {destination}")
        temporary.rename(destination)
        elapsed = time.monotonic() - started
        print(
            f"  Saved {destination.name}: {rows:,} rows, {len(columns)} columns, "
            f"{destination.stat().st_size / 2**20:,.1f} MiB, {elapsed:.1f} seconds",
            flush=True,
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()



def convert_task(source, destination, kind):
    """Top-level worker function so Windows can start a separate process."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {source.parent.name}/{source.name}", flush=True)
    convert_file(source, destination, kind)


def main(workers=2):
    if not EXTRACTED.is_dir():
        raise FileNotFoundError(f"Extracted folder not found: {EXTRACTED}")

    pending = []
    kept = 0
    for folder in sorted(EXTRACTED.iterdir()):
        if not folder.is_dir() or not re.fullmatch(r"[0-9]{4}", folder.name):
            continue
        year = folder.name
        files = []
        for source in sorted(folder.glob("*.txt")):
            match = re.fullmatch(rf"(orig|perf)_{year}Q([1-4])\.txt", source.name)
            if match:
                files.append((source, match.group(1), match.group(2)))
        if not files:
            raise FileNotFoundError(f"No annual loan files found in {folder}")
        pairs = {(kind, quarter) for _, kind, quarter in files}
        for source, kind, quarter in files:
            counterpart = "perf" if kind == "orig" else "orig"
            if (counterpart, quarter) not in pairs:
                raise FileNotFoundError(f"Missing {counterpart}_{year}Q{quarter}.txt")
            destination = PARQUETS / year / (source.stem + ".parquet")
            if not destination.exists():
                pending.append((source, destination, kind))
                continue
            try:
                metadata = pq.read_metadata(destination)
                expected = pa.schema([(name, pa.string()) for name in get_headers(kind)])
                if not metadata.schema.to_arrow_schema().equals(expected):
                    raise ValueError("Unexpected column names or types")
            except Exception as error:
                raise ValueError(
                    f"Existing output cannot be accepted: {destination}\n"
                    "It has not been deleted or overwritten. Review it before restarting."
                ) from error
            kept += 1
            print(f"KEEP {year}/{destination.name} (footer/schema checked)", flush=True)

    if not pending and not kept:
        raise FileNotFoundError(f"No loan datasets found under {EXTRACTED}")
    print(f"\nExisting: {kept} files. Missing: {len(pending)} files.", flush=True)
    if not pending:
        print("Nothing to convert.", flush=True)
        return

    print(f"Parallel workers: {workers}", flush=True)
    if workers == 1:
        for index, task in enumerate(pending, 1):
            convert_task(*task)
            print(f"DONE {index}/{len(pending)}", flush=True)
    else:
        # Only submit as many tasks as there are workers. On failure, no long
        # queue of additional conversions is left running in the background.
        tasks = iter(pending)
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
        ) as pool:
            running = {}
            for _ in range(min(workers, len(pending))):
                task = next(tasks)
                running[pool.submit(convert_task, *task)] = task[0]
            try:
                while running:
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    # Check all finished jobs before submitting more work.
                    for future in finished:
                        future.result()
                    for future in finished:
                        source = running.pop(future)
                        completed += 1
                        print(f"DONE {completed}/{len(pending)}: {source.name}", flush=True)
                        task = next(tasks, None)
                        if task is not None:
                            running[pool.submit(convert_task, *task)] = task[0]
            except BaseException:
                for future in running:
                    future.cancel()
                # Already active workers finish before the pool closes.
                raise
    print(f"\nFinished: {len(pending)} created; {kept} existing files kept.", flush=True)


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    main(workers=args.workers)
