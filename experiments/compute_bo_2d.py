"""Fig 7: one-step BO placement, VM (variance only) vs UCB (mean + kappa*std).

Fixed-hyperparameter diagnostic on a 2D test function. For each initial-design
size N0 and seed, draw a Sobol design, condition a GP (fixed lengthscale and
noise), and record where VM and UCB each place the next point. Aggregating over
seeds gives a selection-frequency density. The contrast isolates the role of
the posterior mean: VM ignores it (stays at the corners regardless of N0),
UCB moves interior as the mean becomes informative.

Writes results/bo_2d.npz.

Usage:
    python compute_bo_2d.py                          # Gramacy-Lee, 10000 seeds
    python compute_bo_2d.py --function peaks
    python compute_bo_2d.py --n-seeds 200            # smoke test
"""
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gp_fit import build_fixed_gp, estimate_noise
from src.test_functions import peaks, grlee08
from src.paths import RESULTS_DIR

FUNCTIONS = {"peaks": peaks, "grlee08": grlee08}

D = 2
N0_VALUES = [5, 10, 25]
N_SEEDS = 10000
LENGTHSCALE = 0.25
KAPPA = 2.0           # UCB = mean + KAPPA * std  (BoTorch beta = KAPPA**2)
NOISE_CV = 0.05
N_GRID = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACQ_NAMES = ["vm", "ucb"]


def candidate_grid(n_grid=N_GRID):
    g = torch.linspace(0.01, 0.99, n_grid, dtype=torch.float64)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(DEVICE)


def acquisitions(model, X_cand, kappa):
    with torch.no_grad():
        pred = model(X_cand)
        var = pred.variance.clamp_min(0)
        mean = pred.mean
    return {"vm": var, "ucb": mean + kappa * var.sqrt()}


def run_one(seed, N0, sigma_y, X_cand, lengthscale, func):
    X_init = SobolEngine(D, scramble=True, seed=seed).draw(N0).double().to(DEVICE)
    with torch.no_grad():
        y_init = func(X_init.cpu()).to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    y_init = y_init + sigma_y * torch.randn(N0, dtype=torch.float64, generator=gen).to(DEVICE)

    model = build_fixed_gp(X_init, y_init, sigma_y, lengthscale, device=DEVICE)
    acq = acquisitions(model, X_cand, KAPPA)
    return {name: X_cand[int(acq[name].argmax())].cpu().numpy() for name in ACQ_NAMES}


def function_surface(func, n=200):
    g = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    X = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        z = func(X)
    return g.numpy(), z.numpy().reshape(n, n)


def main(func_name, n_seeds, lengthscale):
    func = FUNCTIONS[func_name]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X_cand = candidate_grid()
    sigma_y = estimate_noise(func, D, cv=NOISE_CV)
    print(f"Function={func_name}, ell={lengthscale}, kappa={KAPPA}, "
          f"N0={N0_VALUES}, seeds={n_seeds}, sigma_y={sigma_y:.4f}")

    grid_1d, func_surf = function_surface(func)
    selected_x = np.zeros((len(N0_VALUES), len(ACQ_NAMES), n_seeds, D), dtype=np.float64)

    for i_N, N0 in enumerate(N0_VALUES):
        for s in range(n_seeds):
            out = run_one(s, N0, sigma_y, X_cand, lengthscale, func)
            for i_a, name in enumerate(ACQ_NAMES):
                selected_x[i_N, i_a, s] = out[name]
        print(f"  N0={N0:>3} done ({time.time() - t0:.0f}s)", flush=True)

    out_path = RESULTS_DIR / "bo_2d.npz"
    np.savez(
        out_path, selected_x=selected_x, N0_values=np.array(N0_VALUES),
        acq_names=np.array(ACQ_NAMES), function=func_name, kappa=KAPPA,
        lengthscale=lengthscale, sigma_y=sigma_y, n_seeds=n_seeds, n_grid=N_GRID,
        grid_1d=grid_1d, func_surface=func_surf,
    )
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="grlee08", choices=sorted(FUNCTIONS))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--lengthscale", type=float, default=LENGTHSCALE)
    args = parser.parse_args()
    main(args.function, args.n_seeds, args.lengthscale)
