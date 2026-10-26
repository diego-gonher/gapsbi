"""Regression coverage for the non-destructive release-provenance exporter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SPEC = importlib.util.spec_from_file_location(
    "export_release_provenance", Path(__file__).parents[1] / "scripts/export_release_provenance.py"
)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


def test_export_sanitizes_json_and_csv_without_modifying_source(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    source = root / "outputs"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps({"artifact": str(root / "references" / "reference.h5"), "external": "/Users/example/work"}),
        encoding="utf-8",
    )
    (source / "metrics.csv").write_text(
        "source,note\n" + f"{root}/outputs/results.csv,/home/example/scratch\n",
        encoding="utf-8",
    )
    destination = tmp_path / "release"

    assert EXPORT.export_path(source, destination, root) == 2
    assert json.loads((destination / "manifest.json").read_text()) == {
        "artifact": "references/reference.h5", "external": "<local-path>"
    }
    assert (destination / "metrics.csv").read_text(encoding="utf-8") == (
        "source,note\noutputs/results.csv,<local-path>\n"
    )
    assert "/Users/example/work" in (source / "manifest.json").read_text(encoding="utf-8")
