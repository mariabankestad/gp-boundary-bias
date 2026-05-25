"""Shared code for the boundary-variance-inflation paper.

Modules:
  kernels        Matern-5/2 product / isotropic / Neumann kernels on [0,1]^D
  gp             closed-form fixed-hyperparameter GP posterior
  acquisitions   VM/EIG, NIPV, EPIG (function-free) and fast_epig (fitted GP)
  diagnostic     stratified candidates + volume-corrected argmax
  gp_fit         gpytorch GP models for the sequential and BO experiments
  test_functions peaks, grlee08, smooth_box_3d, hartmann_4d
  plot_style     shared matplotlib style, colours, save_fig
  sweep_utils    load lengthscale-sweep summaries
  paths          canonical results / figures directories
"""
