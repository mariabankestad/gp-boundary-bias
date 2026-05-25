"""gpytorch GP models for the sequential-validation and one-step-BO experiments.

Unlike the closed-form diagnostics, these experiments run on real test
functions, so they need posterior means and (for the sequential runs) fitted
hyperparameters:

  * ``FixedIsotropicGP`` / ``build_fixed_gp`` -- isotropic Matern-5/2 with a
    *fixed* lengthscale and noise (one-step BO, Fig 7). No free parameters.

  * ``FittedIsotropicGP`` / ``fit_gp`` -- the same kernel but with the
    lengthscale fitted by MAP under a D-scaled LogNormal prior (Hvarfner et
    al., ICML 2024), noise held fixed (sequential validation, Fig 4).

  * ``estimate_noise`` -- pilot estimate of observation noise as CV * std(f).

The observation noise is set in the standardized output scale; argmax of the
acquisition is invariant to the affine standardization of y.
"""
import math

import torch
import gpytorch
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import LogNormalPrior
from torch.quasirandom import SobolEngine


class FixedIsotropicGP(gpytorch.models.ExactGP):
    """Isotropic Matern-5/2, zero mean, fixed lengthscale and unit outputscale."""

    def __init__(self, train_x, train_y, likelihood, lengthscale):
        super().__init__(train_x, train_y, likelihood)
        base = gpytorch.kernels.MaternKernel(nu=2.5)
        base.lengthscale = lengthscale
        base.raw_lengthscale.requires_grad_(False)
        self.covar_module = gpytorch.kernels.ScaleKernel(base)
        self.covar_module.outputscale = 1.0
        self.covar_module.raw_outputscale.requires_grad_(False)
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class FittedIsotropicGP(gpytorch.models.ExactGP):
    """Isotropic Matern-5/2 with a D-scaled LogNormal lengthscale prior."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        D = train_x.shape[-1]
        base = gpytorch.kernels.MaternKernel(nu=2.5)
        base.register_constraint("raw_lengthscale", Interval(0.05, 0.5))
        self.covar_module = gpytorch.kernels.ScaleKernel(base)
        self.covar_module.outputscale = 1.0
        self.covar_module.raw_outputscale.requires_grad_(False)
        mu_0, sigma_0 = math.sqrt(2.0), math.sqrt(3.0)
        self.register_prior(
            "lengthscale_prior",
            LogNormalPrior(mu_0 + math.log(D) / 2.0, sigma_0),
            lambda m: m.covar_module.base_kernel.lengthscale,
        )
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def _standardize(y_train, sigma_y):
    y_sd = y_train.std().clamp_min(1e-5)
    y_norm = (y_train - y_train.mean()) / y_sd
    noise_norm = (sigma_y / y_sd.item()) ** 2
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=GreaterThan(1e-6))
    likelihood.noise = noise_norm
    likelihood.raw_noise.requires_grad_(False)
    return y_norm, likelihood


def build_fixed_gp(X_train, y_train, sigma_y, lengthscale, device="cpu"):
    """Condition a fixed-hyperparameter GP on (X_train, y_train); no fitting."""
    y_norm, likelihood = _standardize(y_train, sigma_y)
    model = FixedIsotropicGP(X_train, y_norm, likelihood, lengthscale).to(
        dtype=torch.float64, device=device)
    model.eval()
    likelihood.eval()
    return model


def fit_gp(X_train, y_train, sigma_y, device="cpu"):
    """Fit the lengthscale by MAP (L-BFGS); noise stays fixed."""
    D = X_train.shape[-1]
    y_norm, likelihood = _standardize(y_train, sigma_y)
    model = FittedIsotropicGP(X_train, y_norm, likelihood).to(
        dtype=torch.float64, device=device)

    model.covar_module.base_kernel.lengthscale = math.exp(
        math.sqrt(2.0) - 3.0 + math.log(D) / 2.0)

    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.LBFGS(params, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = -mll(model(X_train), y_norm)
        loss.backward()
        return loss

    optimizer.step(closure)
    model.eval()
    likelihood.eval()
    return model


def estimate_noise(func, D, cv=0.05, n_pilot=4096, seed=9999):
    """Pilot estimate of observation noise: cv * std(f) over a Sobol sample."""
    X_pilot = SobolEngine(D, scramble=True, seed=seed).draw(n_pilot).double()
    with torch.no_grad():
        y_pilot = func(X_pilot)
    return cv * y_pilot.std().item()
