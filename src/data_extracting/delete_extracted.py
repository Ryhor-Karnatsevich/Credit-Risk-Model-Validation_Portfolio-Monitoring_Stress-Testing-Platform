"""Delete extracted loan TXT files only; keep ZIPs, Parquets and prepared data.

Run this script after stopping other data pipelines. It lists the space to
reclaim and asks for DELETE before permanently removing any files.
Only files with an existing local ZIP and readable Parquet are eligible.
This checks Parquet metadata, not full value-by-value equivalence.
"""

from pathlib import Path
import re
import stat

import pyarrow.parquet as pq

if __package__:
    from .schema import get_headers
else:
    from schema import get_headers


WORKSPACE = Path(__file__).resolve().parents[2]
RAW = WORKSPACE / "data" / "raw" / "freddie_mac"
EXTRACTED = RAW / "extracted"
PATTERN = re.compile(r"(orig|perf)_((?:19|20)[0-9]{2})Q([1-4])\.txt")


def checked_path(path):
    """Reject redirected paths and anything outside this workspace."""
    path.relative_to(WORKSPACE)
    for part in (path, *path.parents):
        info = part.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"Refusing a linked/junction path: {part}")
        if part == WORKSPACE:
            break
    resolved = path.resolve(strict=True)
    resolved.relative_to(WORKSPACE)
    return resolved


def check_backups(source, kind, year, quarter):
    parquet = checked_path(RAW / "parquets" / year / source.with_suffix(".parquet").name)
    metadata = pq.read_metadata(parquet)
    if metadata.num_rows <= 0 or metadata.schema.names != get_headers(kind):
        raise ValueError(f"Unexpected/empty Parquet: {parquet}")
    archives = [RAW / "zips" / f"historical_data_{year}.zip",
                RAW / "zips" / f"historical_data_{year}Q{quarter}.zip"]
    for archive in archives:
        if archive.exists():
            archive = checked_path(archive)
            if archive.is_file() and archive.stat().st_size > 0:
                return
    raise FileNotFoundError(f"No local ZIP backup for {source.name}")


def main():
    if not EXTRACTED.exists():
        print("No extracted folder; nothing to delete.")
        return
    extracted = checked_path(EXTRACTED)
    planned = []
    for folder in sorted(extracted.iterdir()):
        if not re.fullmatch(r"(?:19|20)[0-9]{2}", folder.name):
            continue
        folder = checked_path(folder)
        if not folder.is_dir():
            continue
        for source in sorted(folder.iterdir()):
            match = PATTERN.fullmatch(source.name)
            if not match or match.group(2) != folder.name:
                continue
            try:
                source = checked_path(source)
                if not source.is_file():
                    continue
                kind, year, quarter = match.groups()
                check_backups(source, kind, year, quarter)
                info = source.stat()
                planned.append((source, kind, year, quarter, info.st_size, info.st_mtime_ns))
                print(f"Delete: {source.relative_to(extracted)} ({info.st_size / 1024**3:.2f} GiB)")
            except (OSError, ValueError) as error:
                print(f"KEEP {source.name}: {error}")
    if not planned:
        print("No eligible TXT files to delete.")
        return
    total = sum(item[4] for item in planned)
    print(f"\n{len(planned)} TXT files; {total / 1024**3:.2f} GiB to reclaim.")
    print("ZIPs, Parquets, prepared data and results will NOT be deleted.")
    print("Stop any extraction/conversion/comparison scripts first.")
    if input("Permanent deletion (not Recycle Bin). Type DELETE to confirm: ").strip() != "DELETE":
        print("Cancelled; nothing deleted.")
        return
    deleted = freed = 0
    for source, kind, year, quarter, size, mtime in planned:
        try:
            source = checked_path(source)
            source.relative_to(extracted)
            check_backups(source, kind, year, quarter)
            info = source.stat()
            if (info.st_size, info.st_mtime_ns) != (size, mtime):
                raise ValueError("TXT changed since confirmation preview")
            source.unlink()
            deleted += 1
            freed += size
            print(f"Deleted: {source.name}", flush=True)
        except (OSError, ValueError) as error:
            print(f"KEEP {source.name}: {error}", flush=True)
    # Only remove now-empty year folders; never recursively delete directories.
    for folder in sorted({item[0].parent for item in planned}):
        try:
            checked_path(folder).rmdir()
        except (OSError, ValueError):
            pass
    print(f"Done: deleted {deleted}/{len(planned)} TXT files ({freed / 1024**3:.2f} GiB).")
    print("Kept ZIPs can be used to extract the TXT files again if needed.")


if __name__ == "__main__":
    main()
