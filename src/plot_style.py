"""Shared matplotlib style for the paper figures.

A single ``apply_style`` (serif, STIX math, thin axes) used by every figure
script, plus the muted-palette ``COLORS`` map, a ``FULL_WIDTH`` page-width
constant, and a ``save_fig`` helper that writes PNG figures.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

FULL_WIDTH = 6.75   # inches, full text width
TEXT_WIDTH = 5.50

_PALETTE = sns.color_palette("muted")
COLORS = {
    "blue": _PALETTE[0],
    "orange": _PALETTE[1],
    "green": _PALETTE[2],
    "red": _PALETTE[3],
    "purple": _PALETTE[4],
    "brown": _PALETTE[5],
    "pink": _PALETTE[6],
    "gray": _PALETTE[7],
}

# Semantic colours for the three acquisitions (VM/NIPV/EPIG).
METHOD_COLORS = {"VM": _PALETTE[0], "NIPV": _PALETTE[3], "EPIG": _PALETTE[1]}


def apply_style():
    """Set matplotlib defaults shared by all figure scripts."""
    sns.set_palette("muted")
    base = 8
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertine", "STIX", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1.5,
        "ytick.labelsize": base - 1.5,
        "legend.fontsize": base - 1.5,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name, out_dir, formats=("png",)):
    """Save ``fig`` as ``name.<fmt>`` in ``out_dir`` for each format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved: " + ", ".join(str(out_dir / f"{name}.{f}") for f in formats))
