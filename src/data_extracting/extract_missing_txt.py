"""Extract one required TXT from a local annual or quarterly Freddie Mac ZIP."""

from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "freddie_mac"


def archive_candidates(source):
    match = re.fullmatch(r"(?:orig|perf)_([0-9]{4})Q([1-4])\.txt", source.name)
    if not match:
        raise ValueError(f"Unexpected source filename: {source.name}")
    year, quarter = match.groups()
    return [ROOT / "zips" / f"historical_data_{year}.zip",
            ROOT / "zips" / f"historical_data_{year}Q{quarter}.zip"]


def copy_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < member.file_size + 1024**3:
        raise OSError(f"Not enough disk space to extract {destination.name} with a 1 GiB reserve")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".extracting", delete=False,
        ) as output:
            temporary = Path(output.name)
            with archive.open(member) as input_file:
                # Reading to the end also verifies this member's ZIP CRC.
                shutil.copyfileobj(input_file, output, length=8 * 1024 * 1024)
        if temporary.stat().st_size != member.file_size:
            raise ValueError(f"Extracted size mismatch: {destination.name}")
        if destination.exists():
            raise FileExistsError(f"Will not overwrite: {destination}")
        temporary.rename(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def find_member(archive, source):
    matches = [item for item in archive.infolist()
               if not item.is_dir() and PurePosixPath(item.filename).name == source.name]
    if len(matches) > 1:
        raise ValueError(f"Archive contains duplicate entries for {source.name}")
    if matches:
        # Never use ZIP paths as output paths; write only the required filename.
        copy_member(archive, matches[0], source)
        return True
    return False


def ensure_source(source):
    if source.is_file():
        return
    candidates = [path for path in archive_candidates(source) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"Missing TXT and local ZIP archive for {source.name}")
    quarter_tag = source.stem.rsplit("_", 1)[-1]
    for path in candidates:
        print(f"Extracting {source.name} from {path.name} ...", flush=True)
        with zipfile.ZipFile(path) as outer:
            if find_member(outer, source):
                return
            nested = [item for item in outer.infolist() if not item.is_dir()
                      and item.filename.lower().endswith(".zip")]
            preferred = [item for item in nested if quarter_tag in item.filename]
            for entry in preferred or nested:
                with outer.open(entry) as stream, zipfile.ZipFile(stream) as inner:
                    if find_member(inner, source):
                        return
    raise FileNotFoundError(f"Local ZIP files do not contain the required {source.name}")
