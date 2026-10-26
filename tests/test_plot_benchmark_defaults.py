"""Release-facing defaults for benchmark plotting."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("plot_benchmark_summary", SCRIPTS / "plot_benchmark_summary.py")
assert SPEC is not None and SPEC.loader is not None
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def test_release_plot_defaults_use_partial_mcar_mnar_only(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["plot_benchmark_summary.py", "--all"])
    args = PLOTTER.parse_args()
    assert args.reference_mode == "partial"
    assert {mechanism for _, mechanism in PLOTTER.selected_jobs(args)} == {"mcar", "mnar"}
