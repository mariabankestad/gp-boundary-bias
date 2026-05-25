"""Fig 5: selection-profile robustness across (D, lengthscale).

4x4 grid: rows = D in {2, 3, 4, 6}, columns = ls in {0.20, 0.30, 0.40, 0.55}.
Each panel shows volume-corrected selection profiles for VM, NIPV, EPIG
(product kernel) against the geometric no-preference reference. VM stays at the
boundary in every setting; NIPV and EPIG sit interior, drifting boundary-ward
only at large ls in low D.

Reads the sweep summary, writes figs/sweep_profiles.png.

Usage:
    python plot_sweep_profiles.py            # N=50
    python plot_sweep_profiles.py --n 20
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plot_style import apply_style
from src.sweep_utils import load_sweep
from src.paths import FIGS_DIR

DIMS = [2, 3, 4, 6]
LS_VALS = [0.20, 0.30, 0.40, 0.55]
D_MIN, D_MAX = 0.01, 0.5
N_SEL_BINS, SMOOTH_SIGMA, LINE_WIDTH = 22, 1.1, 1.6

_P = sns.color_palette("muted")
METHODS_PLOT = [
    ("EIG product", "VM", _P[0], "-"),
    ("NIPV product", "NIPV", _P[3], "--"),
    ("EPIG product", "EPIG", _P[1], "-."),
]
COLOR_REF = "gray"


def selection_profile(vc, n_bins=N_SEL_BINS, sigma=SMOOTH_SIGMA, d_min=D_MIN, d_max=D_MAX):
    edges = np.linspace(d_min, d_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_seeds = len(vc)
    frac = np.histogram(vc, bins=edges)[0].astype(float)
    frac = frac / frac.sum() if frac.sum() > 0 else frac
    sm = np.maximum(gaussian_filter1d(frac, sigma=sigma), 0.0)
    if sm.sum() > 0:
        sm = sm / sm.sum()
    se = np.sqrt(np.maximum(sm * (1 - sm), 0.0) / n_seeds)
    return centers, sm, 1.96 * se


def uniform_reference(D, n_bins=N_SEL_BINS, d_min=D_MIN, d_max=D_MAX):
    edges = np.linspace(d_min, d_max, n_bins + 1)
    vol = np.maximum(1 - 2 * edges[:-1], 0) ** D - np.maximum(1 - 2 * edges[1:], 0) ** D
    vol = np.maximum(vol, 0.0)
    return vol / vol.sum() if vol.sum() > 0 else vol


def main(n_train):
    apply_style()
    fig, axes = plt.subplots(len(DIMS), len(LS_VALS), figsize=(7.5, 5.4),
                             sharex=True, sharey="row",
                             gridspec_kw={"wspace": 0.14, "hspace": 0.42, "left": 0.08,
                                          "right": 0.985, "top": 0.91, "bottom": 0.08})
    row_ymax, plot_data = [], {}
    for D in DIMS:
        d = load_sweep(D)
        methods = [str(m) for m in d["methods"]]
        i_N = list(d["N_trains"]).index(n_train)
        ls_vals = list(d["ls_vals"])
        max_y_nonvm = 0.0
        for i_col, ls in enumerate(LS_VALS):
            i_ls = [j for j, v in enumerate(ls_vals) if abs(v - ls) < 1e-6][0]
            curves = []
            for key, label, color, dash in METHODS_PLOT:
                cx, prof, ci = selection_profile(d["vc_dbnd"][i_N, i_ls, methods.index(key)])
                curves.append((label, color, dash, cx, prof, ci))
                if label != "VM":
                    max_y_nonvm = max(max_y_nonvm, (prof + ci).max())
            uref = uniform_reference(D)
            max_y_nonvm = max(max_y_nonvm, uref.max())
            plot_data[(D, i_col)] = (curves, cx, uref)
        row_ymax.append(min(max_y_nonvm * 1.25, 0.55))

    legend_handles = None
    for r, D in enumerate(DIMS):
        ymax = row_ymax[r]
        for c, ls in enumerate(LS_VALS):
            ax = axes[r, c]
            curves, centers, uref = plot_data[(D, c)]
            ax.yaxis.grid(True, color="#E6E6E6", lw=0.4, ls="-")
            ax.set_axisbelow(True)
            ref_line, = ax.plot(centers, uref, color=COLOR_REF, ls=":", lw=LINE_WIDTH, alpha=0.6, label="No pref.")
            method_lines = []
            for label, color, dash, cx, prof, ci in curves:
                ln, = ax.plot(cx, prof, color=color, ls=dash, lw=LINE_WIDTH, label=label, solid_capstyle="round")
                ax.fill_between(cx, np.maximum(prof - ci, 0), prof + ci, color=color, alpha=0.22, lw=0)
                method_lines.append(ln)
            if legend_handles is None:
                legend_handles = method_lines + [ref_line]
            ax.set_xlim(D_MIN, D_MAX)
            ax.set_ylim(0.0, ymax)
            if r == 0:
                ax.set_title(f"$\\ell = {ls:.2f}$", fontsize=8.5, pad=4)
            if r == len(DIMS) - 1:
                ax.set_xlabel(r"$d_{\partial}$", fontsize=8)
            ax.set_xticks([0.0, 0.25, 0.5])
            ax.set_yticks([0.0, round(ymax * 0.5, 2)])
        axes[r, -1].text(1.04, 0.5, f"$D = {D}$", transform=axes[r, -1].transAxes,
                         rotation=-90, va="center", ha="left", fontsize=9, color="#222222")

    fig.text(0.014, 0.5, "Selection probability", rotation=90, va="center", ha="left",
             fontsize=8.5, color="#222222")
    fig.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False,
               fontsize=8, handlelength=1.6, columnspacing=2.2, borderaxespad=0.0)

    stem = "sweep_profiles" if n_train == 50 else f"sweep_profiles_N{n_train}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / f"{stem}.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / (stem + '.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    main(args.n)
