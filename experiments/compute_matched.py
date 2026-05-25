"""Fig 4 (bottom row): controlled diagnostic at the matched sequential settings.

Runs the function-free EIG/EPIG diagnostic (product Matern-5/2) at the median
hyperparameters fitted in the sequential experiments, so the bottom row of
Fig 4 sits directly under its sequential counterpart:

  D=2: N=35, ls=0.126     D=3: N=40, ls=0.282     D=4: N=50, ls=0.424

Writes results/boundary_profile_matched_D{D}.npz.

Usage:
    python compute_matched.py                       # all configs, 800 seeds
    python compute_matched.py --dims 2 --n-seeds 50  # smoke test
"""
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kernels import matern52_product
from src.acquisitions import compute_eig, compute_epig
from src.diagnostic import make_stratified_query_points
from src.paths import RESULTS_DIR

N_SEEDS = 800
N_PER_BIN = 512
N_BINS = 50
N_TARGET = 2048
SIGMA2 = 0.003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS = ["EIG product", "EPIG product"]
CONFIGS = [
    {"D": 2, "N_train": 35, "ls": 0.126},
    {"D": 3, "N_train": 40, "ls": 0.282},
    {"D": 4, "N_train": 50, "ls": 0.424},
]


def run_config(cfg, n_seeds):
    D, N_train, ls_val = cfg["D"], cfg["N_train"], cfg["ls"]
    print(f"\n{'=' * 60}\n  D={D}, N={N_train}, ls={ls_val}, {n_seeds} seeds\n{'=' * 60}")

    np.random.seed(7777 + D)
    X_q, d_bnd = make_stratified_query_points(
        D, d_min=0.01, n_per_bin=N_PER_BIN, n_bins=N_BINS, device=DEVICE)
    n_query = X_q.shape[0]
    X_tgt = SobolEngine(D, scramble=True, seed=777).draw(N_TARGET).double().to(DEVICE)
    ls = torch.tensor(ls_val, dtype=torch.float64, device=DEVICE)

    acq_raw = np.zeros((len(METHODS), n_seeds, n_query), dtype=np.float32)
    t0 = time.time()
    for s in range(n_seeds):
        gen = torch.Generator().manual_seed(s)
        X_train = torch.rand(N_train, D, dtype=torch.float64, generator=gen).to(DEVICE)
        acq_raw[0, s] = compute_eig(X_q, X_train, ls, matern52_product, SIGMA2).cpu().numpy()
        acq_raw[1, s] = compute_epig(X_q, X_tgt, X_train, ls, matern52_product, SIGMA2).cpu().numpy()
        if (s + 1) % 100 == 0:
            print(f"  seed {s + 1}/{n_seeds} ({time.time() - t0:.0f}s)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"boundary_profile_matched_D{D}.npz"
    np.savez_compressed(
        out_path, acq_raw=acq_raw, d_bnd=d_bnd, N_train=N_train, ls=ls_val,
        methods=np.array(METHODS), sigma2=SIGMA2, n_query=n_query,
        n_seeds=n_seeds, n_per_bin=N_PER_BIN, n_bins=N_BINS, D=D,
    )
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[c["D"] for c in CONFIGS])
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    for cfg in CONFIGS:
        if cfg["D"] in args.dims:
            run_config(cfg, args.n_seeds)
