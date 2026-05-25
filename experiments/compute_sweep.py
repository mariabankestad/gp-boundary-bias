"""Figs 2, 5, 6, 8: lengthscale-sweep summary across (D, N, ls, method).

Draws random training designs and records, for VM/NIPV/EPIG under both the
product Matern-5/2 and the Neumann kernel, two compact summaries per setting:

  * per-stratification-bin mean acquisition value  (drives the acq-value
    profiles, Figs 6 and the App F single-row variant)
  * the volume-corrected argmax boundary distance  (drives the selection
    profiles, Figs 2 and 5, and the Neumann bar chart, Fig 8)

Storing only summaries keeps the artifact small (tens of MB instead of GB).
Writes results/sweep_summary/sweep_summary_D{D}.npz.

Usage:
    python compute_sweep.py                          # all D, seeds 0..1000
    python compute_sweep.py --dims 2 --n-seeds 50    # smoke test
"""
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kernels import matern52_product, neumann_product_normalized
from src.acquisitions import compute_eig, compute_nipv, compute_epig
from src.diagnostic import (
    make_stratified_query_points, per_strat_bin_mean, volume_corrected_argmax_dbnd,
)
from src.paths import SWEEP_DIR

DIMS = [2, 3, 4, 6]
N_TRAINS = [20, 50]
LS_VALS = [0.10, 0.20, 0.30, 0.40, 0.55]
N_SEEDS = 1000
N_PER_BIN = 128
N_BINS = 50
N_TARGET = 2048
SIGMA2 = 0.003
M_IMAGES = 1
D_MIN = 0.01
N_EFF = 10000
VC_RNG_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS = ["EIG product", "EIG Neumann", "NIPV product", "NIPV Neumann",
           "EPIG product", "EPIG Neumann"]


def run_one_D(D, seed_start, seed_end):
    n_query = N_BINS * N_PER_BIN
    n_seeds = seed_end - seed_start
    n_N, n_ls, n_m = len(N_TRAINS), len(LS_VALS), len(METHODS)

    print(f"\n{'=' * 60}\n  D = {D}  seeds [{seed_start}, {seed_end})  "
          f"{n_N} N x {n_ls} ls x {n_m} methods\n{'=' * 60}")

    np.random.seed(7777 + D)
    X_q, d_bnd = make_stratified_query_points(
        D, d_min=D_MIN, n_per_bin=N_PER_BIN, n_bins=N_BINS, device=DEVICE)
    X_int = SobolEngine(D, scramble=True, seed=777).draw(N_TARGET).double().to(DEVICE)

    def kfn_neum(x1, x2, ls):
        return neumann_product_normalized(x1, x2, ls, M_IMAGES)

    # (name, family, kernel_fn)
    method_specs = [
        ("EIG product", "eig", matern52_product), ("EIG Neumann", "eig", kfn_neum),
        ("NIPV product", "nipv", matern52_product), ("NIPV Neumann", "nipv", kfn_neum),
        ("EPIG product", "epig", matern52_product), ("EPIG Neumann", "epig", kfn_neum),
    ]
    assert [m[0] for m in method_specs] == METHODS

    bin_means = np.zeros((n_N, n_ls, n_m, n_seeds, N_BINS), dtype=np.float32)
    vc_dbnd = np.zeros((n_N, n_ls, n_m, n_seeds), dtype=np.float32)

    t0 = time.time()
    for i_N, N_train in enumerate(N_TRAINS):
        for i_ls, ls_val in enumerate(LS_VALS):
            ls = torch.tensor(ls_val, dtype=torch.float64, device=DEVICE)
            K_ic = {matern52_product: matern52_product(X_int, X_q, ls),
                    kfn_neum: kfn_neum(X_int, X_q, ls)}

            acq_chunk = np.zeros((n_m, n_seeds, n_query), dtype=np.float32)
            for si, s in enumerate(range(seed_start, seed_end)):
                gen = torch.Generator().manual_seed(s)
                X_train = torch.rand(N_train, D, dtype=torch.float64, generator=gen).to(DEVICE)
                for i_m, (_, family, kfn) in enumerate(method_specs):
                    if family == "eig":
                        acq = compute_eig(X_q, X_train, ls, kfn, SIGMA2)
                    elif family == "nipv":
                        acq = compute_nipv(X_q, X_int, X_train, ls, kfn, SIGMA2, K_ic[kfn])
                    else:
                        acq = compute_epig(X_q, X_int, X_train, ls, kfn, SIGMA2)
                    acq_chunk[i_m, si, :] = acq.cpu().numpy().astype(np.float32)

            for i_m in range(n_m):
                bin_means[i_N, i_ls, i_m] = per_strat_bin_mean(acq_chunk[i_m], N_PER_BIN, N_BINS)
                vc_dbnd[i_N, i_ls, i_m] = volume_corrected_argmax_dbnd(
                    acq_chunk[i_m], d_bnd, D, n_strat_bins=N_BINS, d_min=D_MIN,
                    n_eff=N_EFF, rng_seed=VC_RNG_SEED)

            print(f"  D={D}  N={N_train:>3}  ls={ls_val:.2f}  done  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    strat_edges = np.linspace(D_MIN, 0.5, N_BINS + 1)
    strat_bin_centers = 0.5 * (strat_edges[:-1] + strat_edges[1:])

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SWEEP_DIR / f"sweep_summary_D{D}.npz"
    np.savez_compressed(
        out_path, bin_means=bin_means, vc_dbnd=vc_dbnd,
        d_bnd=d_bnd.astype(np.float32), strat_bin_centers=strat_bin_centers.astype(np.float32),
        N_trains=np.array(N_TRAINS), ls_vals=np.array(LS_VALS), methods=np.array(METHODS),
        sigma2=SIGMA2, n_query=n_query, n_target=N_TARGET, n_seeds=n_seeds,
        n_bins=N_BINS, n_per_bin=N_PER_BIN, d_min=D_MIN, n_eff=N_EFF,
        vc_rng_seed=VC_RNG_SEED, D=D,
    )
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=DIMS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    t0 = time.time()
    for D in args.dims:
        run_one_D(D, 0, args.n_seeds)
    print(f"\nAll done in {time.time() - t0:.0f}s")
