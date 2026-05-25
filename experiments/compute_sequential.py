"""Fig 4 (top row): one-step sequential acquisition on real test functions.

For each dimension, draws a Sobol initial design, fits a GP (MAP lengthscale,
fixed noise), and records the single point that EIG and EPIG each select. Over
many seeds this gives the boundary-distance distribution of the first
sequential pick, validating that the function-free diagnostic predicts what
full sequential acquisition does.

  D=2: Peaks,        N0=35, grid candidates
  D=3: Smooth-box,   N0=40, Sobol candidates
  D=4: Hartmann-4,   N0=50, Sobol candidates

Writes results/sequential_{D}d.npz.

Usage:
    python compute_sequential.py                       # all dims, 1000 seeds
    python compute_sequential.py --dims 2 --n-seeds 20  # smoke test
"""
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gp_fit import fit_gp, estimate_noise
from src.acquisitions import fast_epig
from src.test_functions import peaks, smooth_box_3d, hartmann_4d
from src.paths import RESULTS_DIR

N_SEEDS = 1000
NOISE_CV = 0.05
N_TARGET = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-dimension setup: function, N0, and candidate set.
CONFIGS = {
    2: dict(func=peaks, N0=35, cand="grid", n_grid=100),
    3: dict(func=smooth_box_3d, N0=40, cand="sobol", n_cand=20000),
    4: dict(func=hartmann_4d, N0=50, cand="sobol", n_cand=20000),
}


def make_candidates(cfg, D):
    if cfg["cand"] == "grid":
        g = torch.linspace(0.01, 0.99, cfg["n_grid"], dtype=torch.float64)
        xx, yy = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(DEVICE)
    return SobolEngine(D, scramble=True, seed=999).draw(cfg["n_cand"]).double().to(DEVICE) * 0.98 + 0.01


def eig_on_model(model, X_cand):
    """EIG = 0.5 * log(1 + var/noise) from the fitted posterior variance."""
    model.eval()
    with torch.no_grad():
        var = model(X_cand).variance.clamp_min(0)
    return 0.5 * torch.log1p(var / model.likelihood.noise.item())


def run_seed(seed, D, cfg, sigma_y, X_cand, X_target):
    func, N0 = cfg["func"], cfg["N0"]
    X_init = SobolEngine(D, scramble=True, seed=seed).draw(N0).double().to(DEVICE)
    with torch.no_grad():
        y_init = func(X_init.cpu()).to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    y_init = y_init + sigma_y * torch.randn(N0, dtype=torch.float64, generator=gen).to(DEVICE)

    out = {"X_init": X_init.cpu().numpy(), "y_init": y_init.cpu().numpy()}
    for acq_name in ["eig", "epig"]:
        model = fit_gp(X_init, y_init, sigma_y, device=DEVICE)
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        noise = model.likelihood.noise.detach().cpu().item()
        with torch.no_grad():
            acq = eig_on_model(model, X_cand) if acq_name == "eig" \
                else fast_epig(model, X_cand, X_target)
        x_new = X_cand[int(acq.argmax())]
        out[f"{acq_name}_selected_x"] = x_new.cpu().numpy().reshape(1, -1)
        out[f"{acq_name}_lengthscales"] = ls
        out[f"{acq_name}_noises"] = np.array([noise])
    return out


def run_dimension(D, n_seeds):
    cfg = CONFIGS[D]
    sigma_y = estimate_noise(cfg["func"], D, cv=NOISE_CV)
    print(f"\nD={D} ({cfg['func'].name}, N0={cfg['N0']}): sigma_y={sigma_y:.4f}")

    X_cand = make_candidates(cfg, D)
    X_target = SobolEngine(D, scramble=True, seed=777).draw(N_TARGET).double().to(DEVICE) * 0.96 + 0.02

    results = {"n0": cfg["N0"], "n_seeds": n_seeds, "sigma_y": sigma_y, "D": D}
    t0 = time.time()
    for s in range(n_seeds):
        for k, v in run_seed(s, D, cfg, sigma_y, X_cand, X_target).items():
            results[f"seed{s}_{k}"] = v
        if (s + 1) % 100 == 0:
            print(f"  seed {s + 1}/{n_seeds} ({time.time() - t0:.0f}s)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"sequential_{D}d.npz"
    np.savez(out, **results)
    print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    for D in args.dims:
        run_dimension(D, args.n_seeds)
