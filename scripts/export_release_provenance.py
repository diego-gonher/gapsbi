"""Copy provenance files for release after removing machine-local paths.

The source is never modified. JSON and CSV values are sanitized recursively;
other files are copied verbatim. Repository-relative paths become relative,
while other ``/Users/...`` and ``/home/...`` paths are replaced by the stable
``<local-path>`` marker.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any


LOCAL_PATH = re.compile(r"(?:/Users|/home)/[^/\s,]+(?:/[^\s,]*)?")


def sanitize_value(value: Any, repository_root: Path) -> Any:
    """Return a recursively sanitized provenance value."""
    if isinstance(value, dict):
        return {key: sanitize_value(item, repository_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, repository_root) for item in value]
    if not isinstance(value, str):
        return value

    root = str(repository_root.resolve())
    if value == root:
        return "."
    value = value.replace(f"{root}/", "")
    return LOCAL_PATH.sub("<local-path>", value)


def _export_file(source: Path, destination: Path, repository_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        destination.write_text(
            json.dumps(sanitize_value(payload, repository_root), indent=2) + "\n",
            encoding="utf-8",
        )
    elif source.suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as read_handle:
            reader = csv.DictReader(read_handle)
            rows = list(reader)
            fields = reader.fieldnames
        if fields is None:
            raise ValueError(f"CSV file has no header: {source}")
        with destination.open("w", newline="", encoding="utf-8") as write_handle:
            writer = csv.DictWriter(write_handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sanitize_value(row, repository_root) for row in rows)
    else:
        shutil.copy2(source, destination)


def export_path(source: Path, destination: Path, repository_root: Path) -> int:
    """Sanitize-copy one file or a directory; return the number of files."""
    source = source.resolve()
    destination = destination.resolve()
    repository_root = repository_root.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if source.is_dir() and source in destination.parents:
        raise ValueError("Output must not be inside the input directory.")
    if source.is_file():
        _export_file(source, destination, repository_root)
        return 1
    count = 0
    for item in source.rglob("*"):
        if item.is_file():
            _export_file(item, destination / item.relative_to(source), repository_root)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Artifact file or directory to export.")
    parser.add_argument("--output", type=Path, required=True, help="New destination; source is never modified.")
    parser.add_argument(
        "--repository-root", type=Path, default=Path.cwd(),
        help="Root used to convert repository-local absolute paths to relative paths (default: cwd).",
    )
    args = parser.parse_args()
    print(f"exported_files={export_path(args.input, args.output, args.repository_root)} output={args.output}")


if __name__ == "__main__":
    main()
