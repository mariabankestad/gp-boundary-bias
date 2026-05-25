"""Fig 4: sequential validation vs the controlled diagnostic.

Top row: boundary-distance histograms of the first sequentially selected point
on Peaks (D=2), Smooth-box (D=3), Hartmann-4 (D=4), for EIG and EPIG. Bottom
row: the volume-weighted selection-probability profile from the controlled
diagnostic at the matched hyperparameters. The close agreement validates the
function-free diagnostic against full sequential acquisition.

Reads results/sequential_{D}d.npz and results/boundary_profile_matched_D{D}.npz.
Writes figs/combined_bias.png.

Usage:
    python plot_combined_bias.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plot_style import apply_style, COLORS, FULL_WIDTH, save_fig
from src.paths import RESULTS_DIR, FIGS_DIR

EIG_COLOR, EPIG_COLOR, UNIFORM_COLOR = COLORS["blue"], COLORS["red"], COLORS["gray"]
N_BOOTSTRAP = 5000


def dist_to_boundary(X):
    return np.minimum(X, 1.0 - X).min(axis=1)


def load_sequential(dim):
    path = RESULTS_DIR / f"sequential_{dim}d.npz"
    return dict(np.load(path, allow_pickle=True)) if path.exists() else None


def load_matched_profile(D):
    path = RESULTS_DIR / f"boundary_profile_matched_D{D}.npz"
    return dict(np.load(path, allow_pickle=True)) if path.exists() else None


def collect_boundary_distances(data, acq_name):
    d_vals = []
    for s in range(100000):
        key = f"seed{s}_{acq_name}_selected_x"
        if key not in data:
            if s == 0:
                continue
            break
        d_vals.extend(dist_to_boundary(data[key]).tolist())
    return np.array(d_vals)


def get_argmax_d(prof_data, method_name):
    acq_raw, d_bnd = prof_data["acq_raw"], prof_data["d_bnd"]
    im = list(prof_data["methods"]).index(method_name)
    n_seeds = acq_raw.shape[1]
    return np.array([d_bnd[acq_raw[im, s].argmax()] for s in range(n_seeds)])


def geometric_bin_weights(bin_edges, D):
    a, b = bin_edges[:-1], bin_edges[1:]
    w = (1 - 2 * a) ** D - (1 - 2 * b) ** D
    return w / w.sum()


def selection_probability(argmax_d, bin_edges, sigma=1.0, weights=None):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n = len(argmax_d)

    def raw_and_smooth(samples):
        counts = np.histogram(samples, bins=bin_edges)[0]
        p = counts / counts.sum()
        if weights is not None:
            p = p * weights
            p = p / p.sum()
        p_s = gaussian_filter1d(p, sigma=sigma, mode="reflect")
        return p, p_s / p_s.sum()

    p_raw, p_hat = raw_and_smooth(argmax_d)
    rng = np.random.default_rng(42)
    boot = np.zeros((N_BOOTSTRAP, len(bin_centers)))
    for i in range(N_BOOTSTRAP):
        _, boot[i] = raw_and_smooth(argmax_d[rng.choice(n, size=n, replace=True)])
    return p_hat, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0), p_raw, bin_centers


def plot_top_row(axes, seq_list):
    hist_bins = np.linspace(0, 0.5, 20)
    for col, (seq_data, D, func_name) in enumerate(seq_list):
        ax = axes[0, col]
        eig_d = collect_boundary_distances(seq_data, "eig")
        epig_d = collect_boundary_distances(seq_data, "epig")
        t_ref = np.linspace(0, 0.5, 200)
        pdf_ref = 2 * D * (1 - 2 * t_ref) ** (D - 1)
        uniform_mean = 1 / (2 * (D + 1))
        ax.hist(eig_d, bins=hist_bins, alpha=0.6, color=EIG_COLOR, density=True,
                edgecolor="white", linewidth=0.5, label=f"EIG (mean={eig_d.mean():.3f})")
        ax.hist(epig_d, bins=hist_bins, alpha=0.6, color=EPIG_COLOR, density=True,
                edgecolor="white", linewidth=0.5, label=f"EPIG (mean={epig_d.mean():.3f})")
        ax.plot(t_ref, pdf_ref, color=UNIFORM_COLOR, linestyle="--",
                label=f"Uniform (mean={uniform_mean:.3f})")
        N0 = int(seq_data.get("n0", 0))
        if col == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"Sequential: {func_name}  ($D={D}$,  $N_0={N0}$)")
        ax.legend(loc="upper right")
        ax.set_xlim(0, 0.5)


def plot_bottom_row(axes, prof_list):
    for col, (prof_data, D) in enumerate(prof_list):
        ax = axes[1, col]
        if prof_data is None:
            ax.text(0.5, 0.5, "Data pending", transform=ax.transAxes, ha="center",
                    va="center", color=UNIFORM_COLOR, fontsize=8)
            ax.set_xlabel("$d_{\\partial}$"); ax.set_xlim(0, 0.5)
            continue
        N_train, ls_val, n_bins = int(prof_data["N_train"]), float(prof_data["ls"]), int(prof_data["n_bins"])
        bin_edges = np.linspace(0, 0.5, n_bins + 1)
        w = geometric_bin_weights(bin_edges, D)
        probs = {}
        for method, color in [("EIG product", EIG_COLOR), ("EPIG product", EPIG_COLOR)]:
            argmax_d = get_argmax_d(prof_data, method)
            p_hat, lo, hi, p_raw, centers = selection_probability(argmax_d, bin_edges, weights=w)
            ax.scatter(centers, p_raw, color=color, s=4, alpha=0.25, zorder=1)
            ax.plot(centers, p_hat, color=color, label=method.replace(" product", ""), linewidth=1.2, zorder=2)
            ax.fill_between(centers, lo, hi, color=color, alpha=0.15, zorder=1)
            probs[method] = p_hat
        p_base = geometric_bin_weights(bin_edges, D)
        ax.plot(centers, p_base, color=UNIFORM_COLOR, linestyle="--", linewidth=0.8, label="Uniform candidates")
        y_top = max(probs["EPIG product"].max() * 2.2, p_base.max() * 1.8)
        eig_peak = probs["EIG product"].max()
        if eig_peak > y_top:
            peak_x = centers[probs["EIG product"].argmax()]
            ax.annotate(f"$\\uparrow$ {eig_peak:.2f}", xy=(peak_x, y_top * 0.92),
                        color=EIG_COLOR, fontsize=7, ha="center")
        ax.set_xlabel("$d_{\\partial}$")
        if col == 0:
            ax.set_ylabel("Selection prob.\\ per bin")
        ax.set_title(f"Selection profile  ($D={D}$,  $N={N_train}$,  $\\ell={ls_val:.3f}$)")
        ax.legend(loc="upper right")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, y_top)


def main():
    apply_style()
    seq_list = [(load_sequential(2), 2, "Peaks"),
                (load_sequential(3), 3, "Smooth-box-3D"),
                (load_sequential(4), 4, "Hartmann-4D")]
    prof_list = [(load_matched_profile(2), 2), (load_matched_profile(3), 3), (load_matched_profile(4), 4)]
    if all(s[0] is None for s in seq_list):
        print("Missing sequential data; run experiments/compute_sequential.py first.")
        return

    w, h = FULL_WIDTH, FULL_WIDTH * 0.55
    fig, axes = plt.subplots(2, 3, figsize=(w, h), sharex="col", sharey="row", layout="constrained")
    plot_top_row(axes, seq_list)
    plot_bottom_row(axes, prof_list)
    save_fig(fig, "combined_bias", out_dir=FIGS_DIR)


if __name__ == "__main__":
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    main()
