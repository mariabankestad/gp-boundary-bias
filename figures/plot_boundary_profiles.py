"""Fig 2: selection profiles per dimension from the sweep summary.

One panel per D in {2, 3, 4, 6}, each showing the volume-corrected selection
probability vs boundary distance for VM, NIPV, EPIG (product kernel) against
the geometric no-preference reference. The lengthscale per D follows the
nearest-neighbour rule (smallest sweep value above N^{-1/D} at N=50).

  --kind sel : selection profiles      -> figs/boundary_profiles_sel.png
  --kind acq : acquisition-value rows  -> figs/boundary_profiles_acq.png

Usage:
    python plot_boundary_profiles.py                 # Fig 2 (sel, N=50)
    python plot_boundary_profiles.py --kind acq
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
LS_PER_D = {2: 0.20, 3: 0.30, 4: 0.40, 6: 0.55}
D_MIN, D_MAX = 0.01, 0.5
N_SEL_BINS, SMOOTH_SIGMA, N_PLOT_BINS_ACQ = 22, 1.1, 30
LINE_WIDTH = 1.6

_P = sns.color_palette("muted")
METHODS_PLOT = [
    ("EIG product", "VM", _P[0], "-"),
    ("NIPV product", "NIPV", _P[3], "--"),
    ("EPIG product", "EPIG", _P[1], "-."),
]
COLOR_REF = "gray"


def selection_profile(vc_dbnd_seeds, n_bins=N_SEL_BINS, sigma=SMOOTH_SIGMA, d_min=D_MIN, d_max=D_MAX):
    edges = np.linspace(d_min, d_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_seeds = len(vc_dbnd_seeds)
    hist = np.histogram(vc_dbnd_seeds, bins=edges)[0].astype(float)
    frac = hist / hist.sum() if hist.sum() > 0 else hist
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


def acq_profile(bin_means_seeds, strat_centers, n_plot_bins=N_PLOT_BINS_ACQ, d_min=D_MIN, d_max=D_MAX):
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


def main(kind, n_train):
    apply_style()
    fig, axes = plt.subplots(1, len(DIMS), figsize=(7.5, 1.5), gridspec_kw={"wspace": 0.22})
    legend_handles = None

    for ci, D in enumerate(DIMS):
        ax = axes[ci]
        d = load_sweep(D)
        methods = [str(m) for m in d["methods"]]
        i_N = list(d["N_trains"]).index(n_train)
        ls = LS_PER_D[D]
        i_ls = [j for j, v in enumerate(d["ls_vals"]) if abs(v - ls) < 1e-6][0]

        if kind == "sel":
            uref = uniform_reference(D)
            cx_ref = np.linspace(D_MIN, D_MAX, N_SEL_BINS + 1)[:-1] + 0.5 * (D_MAX - D_MIN) / N_SEL_BINS
            ref_line, = ax.plot(cx_ref, uref, color=COLOR_REF, ls=":", lw=LINE_WIDTH, alpha=0.6, label="No pref.")
            method_lines = []
            for key, label, color, dash in METHODS_PLOT:
                vc = d["vc_dbnd"][i_N, i_ls, methods.index(key)]
                cx, prof, ci_ = selection_profile(vc)
                ln, = ax.plot(cx, prof, color=color, ls=dash, lw=LINE_WIDTH, label=label, solid_capstyle="round")
                ax.fill_between(cx, np.maximum(prof - ci_, 0), prof + ci_, color=color, alpha=0.12, lw=0)
                method_lines.append(ln)
            ax.set_ylim(0, 0.30)
            ax.set_ylabel("Selection probability" if ci == 0 else "")
        else:
            strat_centers = d["strat_bin_centers"]
            method_lines = []
            for key, label, color, dash in METHODS_PLOT:
                bm = d["bin_means"][i_N, i_ls, methods.index(key)]
                cx, prof, ci_ = acq_profile(bm, strat_centers)
                ln, = ax.plot(cx, prof, color=color, ls=dash, lw=LINE_WIDTH, label=label, solid_capstyle="round")
                ax.fill_between(cx, np.maximum(prof - ci_, 0), prof + ci_, color=color, alpha=0.12, lw=0)
                method_lines.append(ln)
            ref_line = ax.axhline(1.0, color=COLOR_REF, ls=":", lw=LINE_WIDTH, alpha=0.6, label="No pref.")
            ax.set_ylim(0, 4.0)
            ax.set_ylabel("Relative acquisition" if ci == 0 else "")

        if ci != 0:
            ax.set_yticklabels([])
        if legend_handles is None:
            legend_handles = method_lines + [ref_line]
        ax.set_xlim(0, 0.5)
        ax.set_xlabel(r"$d_{\partial}$")
        ax.set_xticks([0.0, 0.25, 0.5])
        ax.set_title(f"$D = {D}$,  $\\ell = {LS_PER_D[D]:.2f}$", fontsize=8, pad=3)

    axes[-1].legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles],
                    loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6.5,
                    handlelength=1.6, borderaxespad=0.0)

    canonical = {"sel": "boundary_profiles_sel", "acq": "boundary_profiles_acq"}[kind]
    stem = canonical if n_train == 50 else f"{canonical}_N{n_train}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / f"{stem}.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / (stem + '.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["sel", "acq"], default="sel")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    main(args.kind, args.n)
