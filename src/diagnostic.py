"""Selection-profile diagnostic: stratified candidates and volume reweighting.

The diagnostic asks where each acquisition's argmax lands relative to the
domain boundary. Two pieces make the answer interpretable:

  * ``make_stratified_query_points`` lays down equal numbers of candidates in
    each boundary-distance bin, so rare near-boundary shells are not
    under-sampled.

  * ``volume_corrected_argmax_dbnd`` then subsamples those candidates back to
    the geometric shell volumes before taking the argmax, recovering the
    behaviour you would see under uniform candidates while keeping the
    near-boundary resolution. This separates an acquisition's *preference*
    from the cube's geometry.

``per_strat_bin_mean`` summarizes mean acquisition value per stratification
bin, used by the acquisition-value-profile figures.
"""
import numpy as np
import torch


def make_stratified_query_points(D, d_min=0.01, n_per_bin=128, n_bins=50,
                                 device="cpu"):
    """Equal candidate count per boundary-distance bin on [d_min, 0.5].

    For each bin centre t, points are drawn uniformly in [t, 1-t]^D and then one
    coordinate is pinned to t (or 1-t), placing the point at boundary distance
    t. Returns ``(X_q, d_bnd)`` with ``X_q`` a (n_bins*n_per_bin, D) tensor on
    ``device`` and ``d_bnd`` the per-point boundary distance min(x, 1-x).
    """
    bin_edges = np.linspace(d_min, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    all_points, all_dbnd = [], []
    for b in range(n_bins):
        t = np.clip(bin_centers[b], 0.001, 0.499)
        lo, hi = t, 1.0 - t
        pts = np.random.uniform(lo, hi, size=(n_per_bin, D))
        dims = np.random.randint(0, D, size=n_per_bin)
        sides = np.random.choice([0, 1], size=n_per_bin)
        for i in range(n_per_bin):
            pts[i, dims[i]] = t if sides[i] == 0 else (1.0 - t)
        all_points.append(pts)
        all_dbnd.append(np.minimum(pts, 1.0 - pts).min(axis=1))

    X = np.concatenate(all_points, axis=0)
    d_bnd = np.concatenate(all_dbnd, axis=0)
    return torch.from_numpy(X).double().to(device), d_bnd


def per_strat_bin_mean(acq_seeds, n_per_bin, n_strat_bins):
    """Mean acquisition value per stratification bin, per seed.

    ``acq_seeds`` is (n_seeds, n_bins*n_per_bin) with candidates concatenated
    in bin order. Returns (n_seeds, n_strat_bins).
    """
    n_seeds = acq_seeds.shape[0]
    reshaped = acq_seeds.reshape(n_seeds, n_strat_bins, n_per_bin)
    return reshaped.mean(axis=2).astype(np.float32)


def volume_corrected_argmax_dbnd(acq_seeds, d_bnd, D, n_strat_bins=50,
                                 d_min=0.01, n_eff=10000, rng_seed=42):
    """Boundary distance of the argmax under volume-reweighted candidates.

    For each seed, subsample the stratified candidates so each bin's count is
    proportional to its geometric shell volume in [d_min, 0.5], then take the
    argmax over the subsample. Returns ``d_bnd`` at the argmax, one value per
    seed. The fixed ``rng_seed`` keeps the subsampling reproducible across
    figures.
    """
    strat_edges = np.linspace(d_min, 0.5, n_strat_bins + 1)
    strat_bin = np.clip(np.digitize(d_bnd, strat_edges) - 1, 0, n_strat_bins - 1)
    vol = np.maximum(1 - 2 * strat_edges[:-1], 0) ** D \
        - np.maximum(1 - 2 * strat_edges[1:], 0) ** D
    vol_frac = vol / vol.sum()
    n_per_bin = np.maximum(np.round(n_eff * vol_frac).astype(int), 1)
    bin_members = [np.where(strat_bin == b)[0] for b in range(n_strat_bins)]

    rng = np.random.RandomState(rng_seed)
    n_seeds = acq_seeds.shape[0]
    out = np.zeros(n_seeds, dtype=np.float32)
    for s in range(n_seeds):
        subsampled = []
        for b in range(n_strat_bins):
            members = bin_members[b]
            if len(members) == 0:
                continue
            n_take = min(n_per_bin[b], len(members))
            subsampled.append(rng.choice(members, size=n_take, replace=False))
        pool = np.concatenate(subsampled)
        out[s] = d_bnd[pool[acq_seeds[s, pool].argmax()]]
    return out
