"""Load lengthscale-sweep summaries, merging any extra-lengthscale chunks.

The main sweep is ``results/sweep_summary/sweep_summary_D{D}.npz``. Extra
lengthscales computed later (via ``compute_sweep.py --ls ...``) land in
``sweep_summary_D{D}_ls*.npz``; ``load_sweep`` merges them into one in-memory
dict sorted by lengthscale.
"""
import numpy as np

from .paths import SWEEP_DIR, require


def load_sweep(D):
    """Return the merged sweep dict for dimension ``D`` (sorted by ls)."""
    main_path = SWEEP_DIR / f"sweep_summary_D{D}.npz"
    require(main_path, "python experiments/compute_sweep.py")

    main = dict(np.load(main_path, allow_pickle=True))
    ls_vals = list(main["ls_vals"])
    bin_means_chunks = [main["bin_means"]]
    vc_dbnd_chunks = [main["vc_dbnd"]]

    for extra in sorted(SWEEP_DIR.glob(f"sweep_summary_D{D}_ls*.npz")):
        ex = np.load(extra, allow_pickle=True)
        ex_ls = list(ex["ls_vals"])
        new_idx = [i for i, v in enumerate(ex_ls)
                   if not any(abs(v - u) < 1e-9 for u in ls_vals)]
        if not new_idx:
            continue
        ls_vals.extend(ex_ls[i] for i in new_idx)
        bin_means_chunks.append(ex["bin_means"][:, new_idx])
        vc_dbnd_chunks.append(ex["vc_dbnd"][:, new_idx])

    bm = np.concatenate(bin_means_chunks, axis=1)
    vd = np.concatenate(vc_dbnd_chunks, axis=1)
    ls_arr = np.asarray(ls_vals)
    order = np.argsort(ls_arr)
    main["bin_means"] = bm[:, order]
    main["vc_dbnd"] = vd[:, order]
    main["ls_vals"] = ls_arr[order]
    return main
