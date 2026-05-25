"""Fig 7: one-step BO placement, VM vs UCB across N0.

2x3 grid of selection-frequency heatmaps (rows = {VM, UCB}, columns = N0).
Densities are plain 2D histograms of selected points (un-symmetrized, since the
test function is not symmetric), with function contours overlaid. VM stays at
the corners for every N0; UCB moves interior as N0 grows and the posterior mean
becomes informative.

Reads results/bo_2d.npz, writes figs/bo_2d.png.

Usage:
    python plot_bo_2d.py
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plot_style import apply_style
from src.paths import RESULTS_DIR, FIGS_DIR, require

ACQ_LABELS = {"vm": "VM", "ucb": "UCB"}
FUNC_DISPLAY = {"peaks": "Peaks", "grlee08": "Gramacy--Lee"}


def density_grid(pts, n_bins, smooth_sigma):
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=n_bins, range=[[0.0, 1.0], [0.0, 1.0]])
    H = gaussian_filter(H.astype(float), sigma=smooth_sigma, mode="constant")
    cx, cy = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    cxx, cyy = np.meshgrid(cx, cy, indexing="ij")
    return cxx, cyy, H


def main(args):
    apply_style()
    require(Path(args.npz), "python experiments/compute_bo_2d.py")
    data = np.load(args.npz, allow_pickle=True)
    selected_x = data["selected_x"]
    N0_values = list(data["N0_values"])
    acq_names = [str(a) for a in data["acq_names"]]
    n_seeds, kappa, ell = int(data["n_seeds"]), float(data["kappa"]), float(data["lengthscale"])
    grid_1d, func_surface = data["grid_1d"], data["func_surface"]
    func_name = str(data["function"]) if "function" in data.files else "peaks"
    n_N, n_acq = len(N0_values), len(acq_names)
    print(f"Loaded {args.npz}: {n_acq} acq x {n_N} N0 x {n_seeds} seeds")

    dens = {}
    for ia, acq in enumerate(acq_names):
        for iN in range(n_N):
            cxx, cyy, H = density_grid(selected_x[iN, ia], args.bins, args.smooth)
            m = H.max()
            dens[(acq, iN)] = (cxx, cyy, H / m if m > 0 else H)

    gxx, gyy = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    f_levels = np.linspace(func_surface.min(), func_surface.max(), 7)[1:-1]

    fig, axes = plt.subplots(n_acq, n_N, figsize=(7.5, 4.3),
                             gridspec_kw={"wspace": 0.12, "hspace": 0.12})
    norm = mcolors.PowerNorm(gamma=args.gamma, vmin=0.0, vmax=1.0)
    last_im = None
    for ia, acq in enumerate(acq_names):
        for iN, N0 in enumerate(N0_values):
            ax = axes[ia, iN]
            cxx, cyy, H = dens[(acq, iN)]
            last_im = ax.pcolormesh(cxx, cyy, H, cmap=args.cmap, shading="auto", norm=norm, rasterized=True)
            ax.contour(gxx, gyy, func_surface, levels=f_levels, colors="white", linewidths=0.5,
                       alpha=0.9, path_effects=[pe.withStroke(linewidth=1.1, foreground="black", alpha=0.55)])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            if ia == 0:
                ax.set_title(f"$N_0 = {N0}$", fontsize=7.5, pad=2)
            if ia == n_acq - 1:
                ax.set_xlabel("$x_1$", fontsize=6.5, labelpad=-1)
            else:
                ax.set_xticklabels([])
            if iN == 0:
                ax.set_ylabel(f"{ACQ_LABELS.get(acq, acq)}\n$x_2$", fontsize=6.5)
            else:
                ax.set_yticklabels([])

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), location="right", fraction=0.046, pad=0.02)
    cbar.set_label("selection freq. (norm.)", fontsize=6, labelpad=2)
    cbar.ax.tick_params(labelsize=5.5)
    fig.suptitle(f"One-step selection on {FUNC_DISPLAY.get(func_name, func_name)}-2D "
                 f"($\\ell={ell}$, $\\kappa={kappa:g}$, {n_seeds:,} designs)", fontsize=7, y=0.99)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / "bo_2d.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / 'bo_2d.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=str(RESULTS_DIR / "bo_2d.npz"))
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--smooth", type=float, default=1.2)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--cmap", default="magma")
    main(parser.parse_args())
