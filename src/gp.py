"""Closed-form GP posterior with fixed hyperparameters.

A zero-mean GP with unit prior variance and a stationary kernel. These
helpers are used by the tutorial notebook and the 1D variance figure to show
the boundary-inflation mechanism directly, without any fitting. The
acquisition diagnostics in ``acquisitions.py`` reimplement the variance and
cross-covariance terms inline for speed over many random designs.
"""
import torch


def posterior(X_test, X_train, y_train, k_func, ls, sigma2, jitter=1e-8):
    """Posterior mean and variance at ``X_test``.

    Inputs are (n, D) tensors; ``y_train`` is (n,). Returns (mean, var), each
    (n_test,). The prior variance is 1, so ``var`` rises toward 1 away from the
    training points and, near a domain boundary, stays higher because the
    kernel's correlation neighbourhood is truncated by the wall.
    """
    N = X_train.shape[0]
    eye = torch.eye(N, dtype=X_train.dtype, device=X_train.device)
    K = k_func(X_train, X_train, ls) + (sigma2 + jitter) * eye
    L = torch.linalg.cholesky(K)

    K_star = k_func(X_test, X_train, ls)                       # (n_test, N)
    W = torch.linalg.solve_triangular(L, K_star.T, upper=False)  # (N, n_test)
    var = (1.0 - W.pow(2).sum(0)).clamp_min(0.0)

    alpha = torch.cholesky_solve(y_train.reshape(-1, 1), L)    # (N, 1)
    mean = (K_star @ alpha).reshape(-1)
    return mean, var


def posterior_variance(X_test, X_train, k_func, ls, sigma2, jitter=1e-8):
    """Posterior variance only (function-free); see ``posterior``."""
    N = X_train.shape[0]
    eye = torch.eye(N, dtype=X_train.dtype, device=X_train.device)
    K = k_func(X_train, X_train, ls) + (sigma2 + jitter) * eye
    L = torch.linalg.cholesky(K)
    K_star = k_func(X_test, X_train, ls)
    W = torch.linalg.solve_triangular(L, K_star.T, upper=False)
    return (1.0 - W.pow(2).sum(0)).clamp_min(0.0)
