"""Variance-driven acquisition functions on a closed-form GP posterior.

Each function takes candidate points, training inputs, a lengthscale, a kernel
function ``k_func(X1, X2, ls)``, and the observation-noise variance ``sigma2``.
The posterior is computed in closed form with unit prior variance and a zero
mean: because these acquisitions depend only on the posterior *covariance*
(not on observed function values), no targets ``y`` are needed. This is the
"function-free" diagnostic the paper builds on.

  VM / EIG : 0.5 * log(1 + s^2(x) / sigma2)              (monotone in variance)
  NIPV     : mean_j cross(x, t_j)^2 / (s^2(x) + sigma2)   (variance reduction)
  EPIG     : mean_j -0.5 * log(1 - rho(x, t_j)^2)         (predictive info gain)

``fast_epig`` is the gpytorch-model variant used in the sequential validation,
where hyperparameters are fitted per seed.
"""
import torch


def compute_eig(X_q, X_train, ls, k_func, sigma2, jitter=1e-5):
    """Expected information gain at each candidate (a.k.a. VM, variance max.)."""
    N = X_train.shape[0]
    eye = torch.eye(N, dtype=X_q.dtype, device=X_q.device)
    K = k_func(X_train, X_train, ls) + (sigma2 + jitter) * eye
    L = torch.linalg.cholesky(K)
    Kqt = k_func(X_q, X_train, ls)
    W = torch.linalg.solve_triangular(L, Kqt.T, upper=False)
    var = (1.0 - W.pow(2).sum(0)).clamp_min(0)
    return 0.5 * torch.log1p(var / sigma2)


def compute_nipv(X_cand, X_int, X_train, ls, k_func, sigma2, K_ic, jitter=1e-5):
    """Negative integrated posterior variance reduction at each candidate.

    ``X_int`` are the integration (target) points and ``K_ic = k(X_int, X_cand)``
    is precomputed because it does not depend on the training design.
    """
    N = X_train.shape[0]
    eye = torch.eye(N, dtype=X_cand.dtype, device=X_cand.device)
    K_TT = k_func(X_train, X_train, ls) + (sigma2 + jitter) * eye
    L = torch.linalg.cholesky(K_TT)

    K_iT = k_func(X_int, X_train, ls)
    W_i = torch.linalg.solve_triangular(L, K_iT.T, upper=False)

    K_cT = k_func(X_cand, X_train, ls)
    W_c = torch.linalg.solve_triangular(L, K_cT.T, upper=False)
    var_c = (1.0 - W_c.pow(2).sum(0)).clamp_min(1e-12) + sigma2

    cross = K_ic - W_i.T @ W_c
    return cross.pow(2).mean(dim=0) / var_c


def compute_epig(X_cand, X_target, X_train, ls, k_func, sigma2, jitter=1e-5):
    """Expected predictive information gain at each candidate."""
    N = X_train.shape[0]
    eye = torch.eye(N, dtype=X_cand.dtype, device=X_cand.device)
    K_TT = k_func(X_train, X_train, ls) + (sigma2 + jitter) * eye
    L = torch.linalg.cholesky(K_TT)

    K_cT = k_func(X_cand, X_train, ls)
    W_c = torch.linalg.solve_triangular(L, K_cT.T, upper=False)
    var_c = (1.0 - W_c.pow(2).sum(0)).clamp_min(1e-12) + sigma2

    K_tT = k_func(X_target, X_train, ls)
    W_t = torch.linalg.solve_triangular(L, K_tT.T, upper=False)
    var_t = (1.0 - W_t.pow(2).sum(0)).clamp_min(1e-12) + sigma2

    K_ct = k_func(X_cand, X_target, ls)
    cross = K_ct - W_c.T @ W_t

    rho2 = (cross.pow(2) / (var_c[:, None] * var_t[None, :])).clamp_max(1 - 1e-6)
    return -0.5 * torch.log1p(-rho2).mean(dim=1)


def vm_nipv_epig(X_cand, X_int, X_train, ls, k_func, sigma2, K_ic, jitter=1e-5):
    """VM, NIPV, and EPIG together, sharing one Cholesky and the solves.

    Used where all three are evaluated under the *same* kernel (the 2D
    surfaces and the matched sequential-validation profiles). ``X_int`` are the
    integration/target points and ``K_ic = k(X_int, X_cand)`` is precomputed.
    Returns ``(vm, nipv, epig)`` each of length ``len(X_cand)``.
    """
    N = X_train.shape[0]
    eye = torch.eye(N, dtype=X_cand.dtype, device=X_cand.device)
    K_TT = k_func(X_train, X_train, ls) + (sigma2 + jitter) * eye
    L = torch.linalg.cholesky(K_TT)

    W_c = torch.linalg.solve_triangular(L, k_func(X_cand, X_train, ls).T, upper=False)
    W_i = torch.linalg.solve_triangular(L, k_func(X_int, X_train, ls).T, upper=False)

    var_c = (1.0 - W_c.pow(2).sum(0)).clamp_min(1e-12)
    var_i = (1.0 - W_i.pow(2).sum(0)).clamp_min(1e-12)

    vm = 0.5 * torch.log1p(var_c / sigma2)

    var_c_noisy = var_c + sigma2
    var_i_noisy = var_i + sigma2
    cross_sq = (K_ic - W_i.T @ W_c).pow(2)

    nipv = cross_sq.mean(dim=0) / var_c_noisy
    rho2 = (cross_sq / (var_c_noisy[None, :] * var_i_noisy[:, None])).clamp_max(1 - 1e-6)
    epig = -0.5 * torch.log1p(-rho2).mean(dim=0)
    return vm, nipv, epig


def fast_epig(model, X_cand, X_target):
    """EPIG on a fitted gpytorch model, avoiding the full joint covariance.

    Memory O(N_c * N_t) instead of O((N_c + N_t)^2). Same result (up to
    floating point) as ``compute_epig`` but reads kernel/noise from the model,
    so it tracks the per-seed fitted hyperparameters in the sequential runs.
    """
    model.eval()
    X_train = model.train_inputs[0]
    N_tr = X_train.shape[0]
    device, dtype = X_cand.device, X_cand.dtype

    noise = model.likelihood.noise.clamp_min(1e-9)
    jitter = 1e-5

    K_tr = model.covar_module(X_train).to_dense()
    K_tr = K_tr + (noise + jitter) * torch.eye(N_tr, device=device, dtype=dtype)
    L = torch.linalg.cholesky(K_tr)

    K_tc = model.covar_module(X_train, X_cand).to_dense()
    K_tt = model.covar_module(X_train, X_target).to_dense()

    W_c = torch.linalg.solve_triangular(L, K_tc, upper=False)
    W_t = torch.linalg.solve_triangular(L, K_tt, upper=False)

    k_cc_diag = model.covar_module(X_cand, diag=True)
    k_tt_diag = model.covar_module(X_target, diag=True)
    var_c = (k_cc_diag - W_c.pow(2).sum(dim=0)).clamp_min(1e-12) + noise
    var_t = (k_tt_diag - W_t.pow(2).sum(dim=0)).clamp_min(1e-12) + noise

    K_ct = model.covar_module(X_cand, X_target).to_dense()
    cross = K_ct - W_c.T @ W_t

    rho_sq = (cross.pow(2) / (var_c.unsqueeze(1) * var_t.unsqueeze(0))).clamp(max=1.0 - 1e-6)
    return -0.5 * torch.log1p(-rho_sq).mean(dim=1)
