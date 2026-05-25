"""Fig 6: acquisition-value robustness across (D, lengthscale).

Same 4x4 layout as Fig 5, but showing the normalized mean acquisition value vs
boundary distance (per-seed normalized so the profile integrates to 0.5). The
mean profiles stay nearly flat around 1 in every panel, in contrast to the
sharp spatial preferences of the argmax in Fig 5: the bias lives in where the
argmax lands, not in the average acquisition magnitude.

Reads the sweep summary, writes figs/acq_sweep.png.

Usage:
    python plot_acq_sweep.py            # N=50
    python plot_acq_sweep.py --n 20
"""
import sys
import argparse
from pathlib import Path

import numpy as np
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
N_PLOT_BINS, LINE_WIDTH = 30, 1.6

_P = sns.color_palette("muted")
METHODS_PLOT = [
    ("EIG product", "VM", _P[0], "-"),
    ("NIPV product", "NIPV", _P[3], "--"),
    ("EPIG product", "EPIG", _P[1], "-."),
]
COLOR_REF = "gray"


def acq_profile(bin_means_seeds, strat_centers, n_plot_bins=N_PLOT_BINS, d_min=D_MIN, d_max=D_MAX):
    edges = np.linspace(d_min, d_max, n_plot_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plot_assign = np.clip(np.digitize(strat_centers, edges) - 1, 0, n_plot_bins - 1)
    n_seeds = bin_means_seeds.shape[0]
    seed_profiles = np.zeros((n_seeds, n_plot_bins), dtype=np.float32)
    for pb in range(n_plot_bins):
        members = np.where(plot_assign == pb)[0]
        if len(members) > 0:
            seed_profiles[:, pb] = bin_means_seeds[:, members].mean(axis=1)
    bin_width = centers[1] - centers[0]
    integral = np.maximum(seed_profiles.sum(axis=1, keepdims=True) * bin_width, 1e-12)
    normed = seed_profiles / integral * 0.5
    return centers, normed.mean(axis=0), 1.96 * normed.std(axis=0) / np.sqrt(n_seeds)


def panel_ylim(curves):
    all_vals = np.concatenate([prof for _, _, _, _, prof, _ in curves])
    all_ci = np.concatenate([ci for _, _, _, _, _, ci in curves])
    half = max(1.0 - float((all_vals - all_ci).min()), float((all_vals + all_ci).max()) - 1.0, 0.15)
    return max(1.0 - half * 1.1, 0.0), 1.0 + half * 1.1


def main(n_train):
    apply_style()
    fig, axes = plt.subplots(len(DIMS), len(LS_VALS), figsize=(7.5, 5.4),
                             sharex=True, sharey=False,
                             gridspec_kw={"wspace": 0.32, "hspace": 0.42, "left": 0.08,
                                          "right": 0.985, "top": 0.91, "bottom": 0.08})
    plot_data = {}
    for D in DIMS:
        d = load_sweep(D)
        methods = [str(m) for m in d["methods"]]
        i_N = list(d["N_trains"]).index(n_train)
        ls_vals = list(d["ls_vals"])
        strat_centers = d["strat_bin_centers"]
        for i_col, ls in enumerate(LS_VALS):
            i_ls = [j for j, v in enumerate(ls_vals) if abs(v - ls) < 1e-6][0]
            curves = []
            for key, label, color, dash in METHODS_PLOT:
                cx, prof, ci = acq_profile(d["bin_means"][i_N, i_ls, methods.index(key)], strat_centers)
                curves.append((label, color, dash, cx, prof, ci))
            plot_data[(D, i_col)] = curves

    legend_handles = None
    for r, D in enumerate(DIMS):
        for c, ls in enumerate(LS_VALS):
            ax = axes[r, c]
            curves = plot_data[(D, c)]
            y_lo, y_hi = panel_ylim(curves)
            ax.yaxis.grid(True, color="#E6E6E6", lw=0.4, ls="-")
            ax.set_axisbelow(True)
            ref_line = ax.axhline(1.0, color=COLOR_REF, ls=":", lw=LINE_WIDTH, alpha=0.6, label="No pref.")
            method_lines = []
            for label, color, dash, cx, prof, ci in curves:
                ln, = ax.plot(cx, prof, color=color, ls=dash, lw=LINE_WIDTH, label=label, solid_capstyle="round")
                ax.fill_between(cx, np.maximum(prof - ci, y_lo), np.minimum(prof + ci, y_hi),
                                color=color, alpha=0.25, edgecolor=color, linewidth=0.5)
                method_lines.append(ln)
            if legend_handles is None:
                legend_handles = method_lines + [ref_line]
            ax.set_xlim(D_MIN, D_MAX)
            ax.set_ylim(y_lo, y_hi)
            if r == 0:
                ax.set_title(f"$\\ell = {ls:.2f}$", fontsize=8.5, pad=4)
            if r == len(DIMS) - 1:
                ax.set_xlabel(r"$d_{\partial}$", fontsize=8)
            ax.set_xticks([0.0, 0.25, 0.5])
            half = (y_hi - y_lo) / 2.0
            step = 0.1 if half < 0.20 else (0.2 if half < 0.50 else 0.5)
            ax.set_yticks([max(round(1.0 - step, 2), 0.0), 1.0, round(1.0 + step, 2)])
        axes[r, -1].text(1.04, 0.5, f"$D = {D}$", transform=axes[r, -1].transAxes,
                         rotation=-90, va="center", ha="left", fontsize=9, color="#222222")

    fig.text(0.014, 0.5, "Relative acquisition", rotation=90, va="center", ha="left",
             fontsize=8.5, color="#222222")
    fig.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False,
               fontsize=8, handlelength=1.6, columnspacing=2.2, borderaxespad=0.0)

    stem = "acq_sweep" if n_train == 50 else f"acq_sweep_N{n_train}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / f"{stem}.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / (stem + '.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    main(args.n)
