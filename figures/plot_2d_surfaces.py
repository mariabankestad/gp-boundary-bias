"""Fig 1: 2x3 panel of acquisition surfaces and argmax-location densities.

Left three panels: VM, NIPV, EPIG surfaces for one fixed Sobol design, with the
argmax marked. Right three panels: argmax-location density over many random
designs, folded into the unit square's fundamental domain (D4 symmetry) and
unfolded for display.

Reads results/surface_2d.npz, writes figs/surface_2d.png.

Usage:
    python plot_2d_surfaces.py
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plot_style import apply_style
from src.paths import RESULTS_DIR, FIGS_DIR, require

PALETTE = sns.color_palette("muted")
METHOD_COLORS = {"VM": PALETTE[0], "NIPV": PALETTE[3], "EPIG": PALETTE[1]}
METHODS = ["VM", "NIPV", "EPIG"]


def to_fundamental(x, y):
    """Map (x, y) in [0,1]^2 to the fundamental domain 0 <= d1 <= d2 <= 0.5."""
    u, v = np.minimum(x, 1 - x), np.minimum(y, 1 - y)
    return np.minimum(u, v), np.maximum(u, v)


def argmax_density_grid(locs, vis_n=200, fund_n=100, sigma=5.0):
    """Density in the fundamental domain, symmetrized and unfolded to [0,1]^2."""
    d1, d2 = to_fundamental(locs[:, 0], locs[:, 1])
    all_a = np.concatenate([d1, d2])
    all_b = np.concatenate([d2, d1])
    H, xe, ye = np.histogram2d(all_a, all_b, bins=fund_n, range=[[0, 0.5], [0, 0.5]])
    H = H.astype(float)
    H = (H + H.T) / 2
    H = gaussian_filter(H, sigma=sigma, mode="reflect")
    H = (H + H.T) / 2
    if H.max() > 0:
        H /= H.max()
    cx, cy = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    interp = RegularGridInterpolator((cx, cy), H, method="linear",
                                     bounds_error=False, fill_value=0.0)
    vis_g = np.linspace(0, 1, vis_n)
    vis_xx, vis_yy = np.meshgrid(vis_g, vis_g, indexing="ij")
    vd1, vd2 = to_fundamental(vis_xx.ravel(), vis_yy.ravel())
    density = interp(np.column_stack([vd1, vd2])).reshape(vis_n, vis_n)
    return vis_xx, vis_yy, density


def main(bottom_cmap, gamma):
    apply_style()
    require(RESULTS_DIR / "surface_2d.npz", "python experiments/compute_2d_surfaces.py")
    data = np.load(RESULTS_DIR / "surface_2d.npz", allow_pickle=True)
    grid = data["grid"]
    surfaces = {"VM": data["vm_surface"], "NIPV": data["nipv_surface"], "EPIG": data["epig_surface"]}
    argmax_pts = {"VM": data["vm_argmax_top"], "NIPV": data["nipv_argmax_top"], "EPIG": data["epig_argmax_top"]}
    X_train = data["X_train_top"]
    argmax_locs = data["argmax_locs"]
    N_seeds = argmax_locs.shape[1]

    fig, axes = plt.subplots(1, 6, figsize=(7.5, 1.5), gridspec_kw={"wspace": 0.15})
    xx, yy = np.meshgrid(grid, grid, indexing="ij")

    bottom = [argmax_density_grid(argmax_locs[ci]) for ci in range(3)]
    shared_vmax = max(d[2].max() for d in bottom)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=shared_vmax)

    im1 = im2 = None
    for ci, mname in enumerate(METHODS):
        color = METHOD_COLORS[mname]
        ax = axes[ci]
        s_normed = surfaces[mname] / surfaces[mname].max()
        im1 = ax.pcolormesh(xx, yy, s_normed, cmap="viridis", shading="auto",
                            rasterized=True, vmin=0, vmax=1)
        ax.plot(X_train[:, 0], X_train[:, 1], "w.", ms=2.5, markeredgecolor="k",
                markeredgewidth=0.3, zorder=5)
        ax.plot(argmax_pts[mname][0], argmax_pts[mname][1], marker="*", color=PALETTE[3],
                ms=8, markeredgecolor="k", markeredgewidth=0.4, zorder=10, clip_on=False)
        ax.set_title(mname, fontsize=7.5, color=color, fontweight="bold", pad=0.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("$x_1$", fontsize=6.5, labelpad=-1)
        if ci == 0:
            ax.set_ylabel("$x_2$", fontsize=6.5)
        else:
            ax.set_yticklabels([])

        ax2 = axes[3 + ci]
        dens_xx, dens_yy, density = bottom[ci]
        im2 = ax2.pcolormesh(dens_xx, dens_yy, density, cmap=bottom_cmap,
                             shading="auto", rasterized=True, norm=norm)
        ax2.set_title(mname, fontsize=7.5, color=color, fontweight="bold", pad=0.2)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect("equal")
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xlabel("$x_1$", fontsize=6.5, labelpad=-1)
        ax2.set_yticklabels([])

    mid_left = (axes[0].get_position().x0 + axes[2].get_position().x1) / 2
    mid_right = (axes[3].get_position().x0 + axes[5].get_position().x1) / 2
    fig.text(mid_left, 0.87, "Acquisition surface (single design)", ha="center",
             va="bottom", fontsize=6.5, fontstyle="italic")
    fig.text(mid_right, 0.87, f"Argmax density ({N_seeds:,} designs)", ha="center",
             va="bottom", fontsize=6.5, fontstyle="italic")

    cbar_w, cbar_h = 0.18, 0.03
    cax1 = fig.add_axes([axes[2].get_position().x1 - cbar_w, 0.01, cbar_w, cbar_h])
    cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
    cb1.set_label("Acq. (norm.)", fontsize=6, labelpad=1); cb1.ax.tick_params(labelsize=5.5, pad=1)
    cax2 = fig.add_axes([axes[5].get_position().x1 - cbar_w, 0.01, cbar_w, cbar_h])
    cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_label("Sel. density", fontsize=6, labelpad=1); cb2.ax.tick_params(labelsize=5.5, pad=1)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / "surface_2d.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / 'surface_2d.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bottom-cmap", default="magma")
    parser.add_argument("--gamma", type=float, default=0.4)
    args = parser.parse_args()
    main(args.bottom_cmap, args.gamma)
