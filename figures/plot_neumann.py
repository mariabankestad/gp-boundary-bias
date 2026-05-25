"""Fig 8: Neumann-correction bar chart.

For each D, the percentage of selections within d < 0.05 of the boundary
(volume-corrected), for: the geometric no-preference baseline, VM with the
standard product kernel, VM with the Neumann kernel, NIPV, and EPIG. The
Neumann kernel reduces VM's boundary concentration, with the reduction growing
in D, but does not eliminate it.

Reads the sweep summary, writes figs/neumann_correction.png.

Usage:
    python plot_neumann.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plot_style import apply_style
from src.sweep_utils import load_sweep
from src.paths import FIGS_DIR

DIMS = [2, 3, 4, 6]
LS_FOR_D = {2: 0.20, 3: 0.30, 4: 0.40, 6: 0.55}   # NN-distance rule at N=50
N_TRAIN, THRESH, D_MIN = 50, 0.05, 0.01

COLOR_UNIFORM, COLOR_VM, COLOR_VM_NEU = "#B0B0B0", "#1F3A68", "#7FA4CC"
COLOR_NIPV, COLOR_EPIG, BAR_ALPHA = "#B23A48", "#D88C2C", 0.93

# (legend_label, color, method_name_in_sweep); None method -> analytical baseline.
BARS = [
    ("no-preference baseline", COLOR_UNIFORM, None),
    ("VM (standard)", COLOR_VM, "EIG product"),
    ("VM (Neumann)", COLOR_VM_NEU, "EIG Neumann"),
    ("NIPV", COLOR_NIPV, "NIPV product"),
    ("EPIG", COLOR_EPIG, "EPIG product"),
]


def uniform_baseline_pct(D, thresh=THRESH, d_min=D_MIN):
    """Geometric P(d < thresh | d in [d_min, 0.5]) under uniform sampling."""
    return 100.0 * (1 - ((1 - 2 * thresh) / (1 - 2 * d_min)) ** D)


def get_bar_value(method, D):
    if method is None:
        return uniform_baseline_pct(D)
    data = load_sweep(D)
    methods = [str(m) for m in data["methods"]]
    i_N = list(data["N_trains"]).index(N_TRAIN)
    i_ls = [j for j, v in enumerate(data["ls_vals"]) if abs(v - LS_FOR_D[D]) < 1e-6][0]
    vc_db = data["vc_dbnd"][i_N, i_ls, methods.index(method)]
    return 100.0 * float((vc_db < THRESH).mean())


def main():
    apply_style()
    values = [[get_bar_value(method, D) for D in DIMS] for _, _, method in BARS]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.6))
    n_bars = len(BARS)
    x = np.arange(len(DIMS), dtype=float)
    bar_w = 0.78 / n_bars
    edge_kw = dict(edgecolor="white", linewidth=0.5)

    bar_groups = []
    for bi, (label, color, _) in enumerate(BARS):
        offset = (bi - (n_bars - 1) / 2) * bar_w
        bar_groups.append(ax.bar(x + offset, values[bi], bar_w, color=color,
                                 alpha=BAR_ALPHA, label=label, **edge_kw))

    for bars in bar_groups:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, max(h, 0) + 1.6,
                    "0" if h < 0.5 else f"{h:.0f}", ha="center", va="bottom",
                    fontsize=6, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$D = {D}$" for D in DIMS])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_ylabel(r"% of selections with $d_\partial < 0.05$", fontsize=8, labelpad=4)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, color="#D9D9D9", linestyle="-", linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.5)

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=5, frameon=False,
                    handlelength=1.1, handleheight=0.9, columnspacing=1.4, fontsize=7, borderaxespad=0.0)
    for patch in leg.get_patches():
        patch.set_alpha(BAR_ALPHA)
        patch.set_edgecolor("none")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / "neumann_correction.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / 'neumann_correction.png'}")


if __name__ == "__main__":
    main()
