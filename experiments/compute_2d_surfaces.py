"""Fig 1: 2D acquisition surfaces and argmax-location density.

Top row: VM, NIPV, EPIG surfaces on a fine grid for one fixed Sobol design.
Bottom row: argmax locations over many random training designs (D4-symmetric
candidate/integration grids, so the density can be folded into the unit
square's fundamental domain at plot time).

Writes results/surface_2d.npz.

Usage:
    python compute_2d_surfaces.py                 # full run, 10000 seeds
    python compute_2d_surfaces.py --n-seeds 200   # smoke test
"""
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kernels import matern52_isotropic
from src.acquisitions import vm_nipv_epig
from src.paths import RESULTS_DIR

D = 2
N_TRAIN = 15
LS = 0.2
SIGMA2 = 0.003
N_GRID = 200
N_INT = 2048
N_SEEDS = 10000
TOP_ROW_SEED = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(n_seeds):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64
    ls = torch.tensor(LS, dtype=dtype, device=DEVICE)

    # Fine grid for the top-row surfaces.
    g = torch.linspace(0.005, 0.995, N_GRID, dtype=dtype, device=DEVICE)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    X_grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    X_int = SobolEngine(D, scramble=True, seed=777).draw(N_INT).to(dtype).to(DEVICE)

    # Top row: one fixed Sobol training design.
    print("Computing top-row surfaces...")
    X_train_top = SobolEngine(D, scramble=True, seed=TOP_ROW_SEED).draw(N_TRAIN).to(dtype).to(DEVICE)
    K_ic = matern52_isotropic(X_int, X_grid, ls)
    vm, nipv, epig = vm_nipv_epig(X_grid, X_int, X_train_top, ls, matern52_isotropic, SIGMA2, K_ic)

    surfaces = {k: v.cpu().numpy().reshape(N_GRID, N_GRID)
                for k, v in [("vm", vm), ("nipv", nipv), ("epig", epig)]}
    argmax_top = {"vm": X_grid[vm.argmax()].cpu().numpy(),
                  "nipv": X_grid[nipv.argmax()].cpu().numpy(),
                  "epig": X_grid[epig.argmax()].cpu().numpy()}
    for k in argmax_top:
        print(f"  {k} argmax: {argmax_top[k]}")

    # Bottom row: argmax over many random designs on a D4-symmetric grid.
    print(f"Computing bottom-row argmax density ({n_seeds} seeds)...")
    argmax_locs = np.zeros((3, n_seeds, 2), dtype=np.float32)

    g_cand = torch.linspace(0.005, 0.995, 128, dtype=dtype, device=DEVICE)
    xx_c, yy_c = torch.meshgrid(g_cand, g_cand, indexing="ij")
    X_cand = torch.stack([xx_c.reshape(-1), yy_c.reshape(-1)], dim=-1)

    g_int = torch.linspace(0.005, 0.995, 45, dtype=dtype, device=DEVICE)
    xx_i, yy_i = torch.meshgrid(g_int, g_int, indexing="ij")
    X_int_sym = torch.stack([xx_i.reshape(-1), yy_i.reshape(-1)], dim=-1)

    K_ic_cand = matern52_isotropic(X_int_sym, X_cand, ls)

    t0 = time.time()
    for s in range(n_seeds):
        gen = torch.Generator().manual_seed(s)
        X_train = torch.rand(N_TRAIN, D, dtype=dtype, generator=gen).to(DEVICE)
        vm_s, nipv_s, epig_s = vm_nipv_epig(X_cand, X_int_sym, X_train, ls, matern52_isotropic, SIGMA2, K_ic_cand)
        argmax_locs[0, s] = X_cand[vm_s.argmax()].cpu().numpy()
        argmax_locs[1, s] = X_cand[nipv_s.argmax()].cpu().numpy()
        argmax_locs[2, s] = X_cand[epig_s.argmax()].cpu().numpy()
        if (s + 1) % 500 == 0:
            print(f"  {s + 1}/{n_seeds} seeds ({time.time() - t0:.0f}s)", flush=True)

    out_path = RESULTS_DIR / "surface_2d.npz"
    np.savez_compressed(
        out_path,
        grid=g.cpu().numpy(),
        vm_surface=surfaces["vm"], nipv_surface=surfaces["nipv"], epig_surface=surfaces["epig"],
        X_train_top=X_train_top.cpu().numpy(),
        vm_argmax_top=argmax_top["vm"], nipv_argmax_top=argmax_top["nipv"], epig_argmax_top=argmax_top["epig"],
        argmax_locs=argmax_locs,
        ls=LS, sigma2=SIGMA2, N_train=N_TRAIN, N_grid=N_GRID, N_int=N_INT,
        N_seeds=n_seeds, top_row_seed=TOP_ROW_SEED,
    )
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    main(args.n_seeds)
