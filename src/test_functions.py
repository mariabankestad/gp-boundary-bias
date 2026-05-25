"""Synthetic test functions used in the sequential and BO experiments.

Each function maps inputs on the unit cube [0,1]^D to its native domain and
returns a (N,) tensor. Native domains are attached as ``.x_min`` / ``.x_max``
metadata. Only the four functions used in the paper are included here:

  peaks         (2D)  MATLAB "peaks" surface on [-3, 3]^2
  grlee08       (2D)  Gramacy & Lee (2008) on [-2, 6]^2
  smooth_box_3d (3D)  product of smooth step functions on [-3, 3]^3
  hartmann_4d   (4D)  Hartmann-4 (Picheny et al. 2012) on [0, 1]^4
"""
import torch


def with_metadata(**kwargs):
    def deco(fn):
        fn.__dict__.update(kwargs)
        return fn
    return deco


def _minmax_to_domain(U, lo, hi):
    """Map U in [0,1] to X in [lo, hi] elementwise."""
    lo = torch.as_tensor(lo, device=U.device, dtype=U.dtype)
    hi = torch.as_tensor(hi, device=U.device, dtype=U.dtype)
    return lo + U * (hi - lo)


@with_metadata(name="grlee08", x_min=[-2.0, -2.0], x_max=[6.0, 6.0],
               expects_unit_cube=True, ref="Gramacy & Lee (2008)")
def grlee08(U):
    """f(x1, x2) = x1 * exp(-x1^2 - x2^2) on [-2, 6]^2; expects U in [0,1]^2."""
    if U.ndim == 1:
        U = U.reshape(1, -1)
    assert U.shape[-1] == 2, "grlee08 expects 2D inputs"
    X = _minmax_to_domain(U, lo=grlee08.x_min, hi=grlee08.x_max)
    x1, x2 = X[..., 0], X[..., 1]
    return x1 * torch.exp(-(x1 ** 2 + x2 ** 2))


@with_metadata(name="peaks", x_min=[-3.0, -3.0], x_max=[3.0, 3.0],
               expects_unit_cube=True, ref="MATLAB 'peaks' test surface")
def peaks(U, a=3.0, b=10.0, c=1.0 / 3.0):
    """MATLAB peaks surface on [-3, 3]^2; expects U in [0,1]^2."""
    if U.ndim == 1:
        U = U.reshape(1, -1)
    assert U.shape[-1] == 2, "peaks expects 2D inputs"
    X = _minmax_to_domain(U, lo=peaks.x_min, hi=peaks.x_max)
    x, y = X[..., 0], X[..., 1]
    return (
        a * (1.0 - x).pow(2) * torch.exp(-(x.pow(2)) - (y + 1.0).pow(2))
        - b * (x / 5.0 - x.pow(3) - y.pow(5)) * torch.exp(-x.pow(2) - y.pow(2))
        - c * torch.exp(-(x + 1.0).pow(2) - y.pow(2))
    )


def _smooth_step_1d(X, box_start=-0.5, box_end=None, k=20):
    rise = torch.sigmoid(k * (X - box_start))
    if box_end:
        fall = 1 - torch.sigmoid(k * (X - box_end))
        return rise * fall
    return rise


@with_metadata(name="smooth_box_3d", x_min=[-3.0, -3.0, -3.0],
               x_max=[3.0, 3.0, 3.0], expects_unit_cube=True,
               ref="Smooth 3D box step function")
def smooth_box_3d(U, k=20, box_start=-0.5, box_end=None):
    """Product of smooth step functions on [-3, 3]^3; expects U in [0,1]^3."""
    if U.ndim == 1:
        U = U.reshape(1, -1)
    X = _minmax_to_domain(U, lo=smooth_box_3d.x_min, hi=smooth_box_3d.x_max)
    bx = _smooth_step_1d(X[:, 0], box_start, box_end, k)
    by = _smooth_step_1d(X[:, 1], box_start, box_end, k)
    bz = _smooth_step_1d(X[:, 2], box_start, box_end, k)
    return bx * by * bz


@with_metadata(name="hartmann_4d", x_min=[0.0] * 4, x_max=[1.0] * 4,
               expects_unit_cube=True,
               ref="Hartmann-4 (Picheny et al. 2012)")
def hartmann_4d(U):
    """Hartmann-4 function (Picheny et al. 2012 rescaling); U in [0,1]^4."""
    if U.ndim == 1:
        U = U.reshape(1, -1)
    assert U.shape[-1] == 4
    X = _minmax_to_domain(U, lo=hartmann_4d.x_min, hi=hartmann_4d.x_max)

    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=U.dtype, device=U.device)
    A = torch.tensor([
        [10.0, 3.0, 17.0, 3.5],
        [0.05, 10.0, 17.0, 0.1],
        [3.0, 3.5, 1.7, 10.0],
        [17.0, 8.0, 0.05, 10.0],
    ], dtype=U.dtype, device=U.device)
    P = torch.tensor([
        [0.1312, 0.1696, 0.5569, 0.0124],
        [0.2329, 0.4135, 0.8307, 0.3736],
        [0.2348, 0.1451, 0.3522, 0.2883],
        [0.4047, 0.8828, 0.8732, 0.5743],
    ], dtype=U.dtype, device=U.device)

    inner = A * (X.unsqueeze(1) - P) ** 2          # (batch, 4, 4)
    terms = alpha * torch.exp(-inner.sum(dim=2))   # (batch, 4)
    outer = terms.sum(dim=1)                        # (batch,)
    return (1.1 - outer) / 0.839
