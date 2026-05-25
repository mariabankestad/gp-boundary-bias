"""Fig 3: 1D posterior variance illustrating boundary inflation.

The posterior variance s^2(x) under a Matern-5/2 kernel with five evenly
spaced interior training points: low near the points, rising toward the prior
between them, and climbing back up near the boundaries x=0 and x=1, where the
kernel's correlation neighbourhood is truncated by the wall. No compute step.

Writes figs/variance_1d.png.

Usage:
    python plot_variance_1d.py
"""
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kernels import matern52_isotropic
from src.gp import posterior_variance
from src.plot_style import apply_style
from src.paths import FIGS_DIR


def main():
    apply_style()
    palette = sns.color_palette("muted")
    color_line, color_prior, color_points, shade_bnd = palette[0], "#9C9C9C", "#222222", "#F5E6E6"

    ls, sigma2 = 0.12, 0.003
    x_train = torch.tensor([0.16, 0.33, 0.50, 0.67, 0.84], dtype=torch.float64).reshape(-1, 1)
    x_grid = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64).reshape(-1, 1)
    s2 = posterior_variance(x_grid, x_train, matern52_isotropic, ls, sigma2).numpy()
    xg = x_grid.numpy().ravel()
    xt = x_train.numpy().ravel()

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 2.0))

    band = 0.08
    ax.axvspan(0.0, band, color=shade_bnd, alpha=0.6, lw=0)
    ax.axvspan(1.0 - band, 1.0, color=shade_bnd, alpha=0.6, lw=0)
    ax.axhline(1.0, color=color_prior, ls="--", lw=0.7, alpha=0.85, label="prior variance")
    ax.plot(xg, s2, color=color_line, lw=1.6, label=r"posterior $s^2(x)$")
    ax.scatter(xt, np.full_like(xt, -0.04), s=22, marker="^", color=color_points,
               clip_on=False, zorder=5, label="training points")

    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.06, 1.15)
    ax.set_xlabel(r"$x$", labelpad=1)
    ax.set_ylabel("variance", labelpad=4)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.4, columnspacing=1.6, fontsize=7, borderaxespad=0.0)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / "variance_1d.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / 'variance_1d.png'}")


if __name__ == "__main__":
    main()
