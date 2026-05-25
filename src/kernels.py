"""Matern-5/2 kernels on [0,1]^D: product, isotropic, and Neumann-corrected.

All kernels take (X1, X2, ls) and return the (n1, n2) Gram matrix. The
lengthscale ls is a scalar (shared across dimensions). The Neumann kernel
adds mirror images across the cube walls so the correlation neighbourhood is
no longer truncated at the boundary; it is the partial correction studied in
the paper.

These are the exact kernels used to produce the figures. The product and
isotropic forms differ for D > 1 even at the same ls, so they are kept
separate: the 2D surfaces (Fig 1) use the isotropic form, while the
selection-profile sweeps use the product form.
"""
import math

import torch


def matern52_1d(r, ls):
    """1D Matern-5/2 correlation as a function of the raw difference r."""
    d = torch.abs(r) / ls
    s5 = math.sqrt(5.0)
    return (1.0 + s5 * d + (5.0 / 3.0) * d.pow(2)) * torch.exp(-s5 * d)


def matern52_product(x1, x2, ls):
    """Product of 1D Matern-5/2 factors, one per dimension."""
    K = torch.ones(x1.shape[0], x2.shape[0], dtype=x1.dtype, device=x1.device)
    for d in range(x1.shape[-1]):
        r = x1[:, d:d + 1] - x2[:, d:d + 1].T
        K = K * matern52_1d(r, ls)
    return K


def matern52_isotropic(X1, X2, ls):
    """Isotropic Matern-5/2 with shared lengthscale (Euclidean distance)."""
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)
    r = diff.norm(dim=-1) / ls
    s5r = math.sqrt(5.0) * r
    return (1.0 + s5r + s5r.pow(2) / 3.0) * torch.exp(-s5r)


# Neumann (mirror-image) kernel

def _neumann_1d_matrix(x1_d, x2_d, ls, M=1):
    diff = x1_d.unsqueeze(1) - x2_d.unsqueeze(0)
    summ = x1_d.unsqueeze(1) + x2_d.unsqueeze(0)
    s = torch.zeros_like(diff)
    for n in range(-M, M + 1):
        s = s + matern52_1d(diff + 2.0 * n, ls) + matern52_1d(summ + 2.0 * n, ls)
    return s


def _neumann_1d_diag(x_d, ls, M=1):
    s = torch.zeros_like(x_d)
    for n in range(-M, M + 1):
        s = s + matern52_1d(torch.full_like(x_d, 2.0 * n), ls)
        s = s + matern52_1d(2.0 * x_d + 2.0 * n, ls)
    return s


def neumann_product_normalized(x1, x2, ls, M=1):
    """Per-dimension Neumann Matern-5/2, normalized to a unit diagonal.

    Each dimension's kernel completes the truncated correlation neighbourhood
    by summing reflections across the walls at 0 and 1 (M images per side),
    then normalizes so k(x, x) = 1. Inputs must lie in [0, 1]^D.
    """
    K = torch.ones(x1.shape[0], x2.shape[0], dtype=x1.dtype, device=x1.device)
    for d in range(x1.shape[-1]):
        s = _neumann_1d_matrix(x1[:, d], x2[:, d], ls, M)
        d1 = _neumann_1d_diag(x1[:, d], ls, M)
        d2 = _neumann_1d_diag(x2[:, d], ls, M)
        K = K * s / torch.sqrt(d1[:, None] * d2[None, :])
    return K
