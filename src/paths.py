"""Canonical directories for the release.

Experiment scripts write computed ``.npz`` arrays under ``RESULTS_DIR`` and
figure scripts read from there and write rendered figures to ``FIGS_DIR``.
Both live at the release root so the two script folders share them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SWEEP_DIR = RESULTS_DIR / "sweep_summary"
FIGS_DIR = ROOT / "figs"


def require(path, command):
    """Stop with a clear message (no traceback) if a results file is missing.

    Plot scripts call this before loading, so running a plot before its compute
    step prints which command to run rather than a bare ``FileNotFoundError``.
    """
    if not Path(path).exists():
        raise SystemExit(
            f"Missing results file: {path}\n"
            f"Run the compute step first:  {command}")
