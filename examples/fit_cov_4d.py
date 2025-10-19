"""Fit 4D covariance matrix to 2D measurements."""

import argparse
import math
import os
import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import ment

plt.style.use("./style.mplstyle")


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dist", type=str, default="gaussian-mixture")
parser.add_argument("--nmeas", type=int, default=10)
parser.add_argument("--xmax", type=float, default=4.0)
parser.add_argument("--bins", type=int, default=80)
parser.add_argument("--nsamp", type=int, default=1000)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--iters", type=int, default=1000)
parser.add_argument("--method", type=str, default="differential_evolution")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Source distribution
# --------------------------------------------------------------------------------------

ndim = 4
dist = ment.dist.get_dist(args.dist, ndim=ndim, seed=args.seed)
x_true = dist.sample(1_000_000)

cov_matrix = torch.cov(x_true.T)
print(cov_matrix)


# Forward model
# --------------------------------------------------------------------------------------

phase_advances = ment.utils.random_uniform(0.0, np.pi, size=(args.nmeas, 2))

transfer_matrices = []
for mux, muy in phase_advances:
    matrix = torch.eye(ndim)
    matrix[0:2, 0:2] = ment.utils.rotation_matrix(mux)
    matrix[2:4, 2:4] = ment.utils.rotation_matrix(muy)
    transfer_matrices.append(matrix)

transforms = []
for matrix in transfer_matrices:
    transform = ment.LinearTransform(matrix)
    transforms.append(transform)

bin_edges = 2 * [torch.linspace(-args.xmax, args.xmax, args.bins + 1)]

diagnostics = []
for transform in transforms:
    diagnostic = ment.HistogramND(axis=(0, 2), edges=bin_edges)
    diagnostics.append([diagnostic])


# Data
# --------------------------------------------------------------------------------------

projections = ment.simulate(x_true, transforms, diagnostics)


# Fit covariance matrix
# --------------------------------------------------------------------------------------

# Run optimizer
fitter = ment.CholeskyCovFitter(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    nsamp=args.nsamp,
    bound=1.00e02,
    verbose=True,
)
cov_matrix, fit_result = fitter.fit(method=args.method, iters=args.iters)


# Print results
print(fit_result)
print(fitter.build_cov())


# Plot results
x = fitter.sample(100_000)
projections_pred = ment.unravel(ment.simulate(x, fitter.transforms, fitter.diagnostics))
projections_true = ment.unravel(fitter.projections)

ncols = min(args.nmeas, 7)
nrows = int(np.ceil(args.nmeas / ncols))
fig, axs = plt.subplots(
    ncols=ncols,
    nrows=nrows,
    figsize=(1.1 * ncols, 1.1 * nrows),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
for proj_true, proj_pred, ax in zip(projections_true, projections_pred, axs.flat):
    ax.pcolormesh(proj_true.coords[0], proj_true.coords[1], proj_true.values.T)
    ax.set_xticks([])
    ax.set_yticks([])

    for i, proj in enumerate([proj_true, proj_pred]):
        color = ["white", "red"][i]
        ls = ["-", "-"][i]

        cx, cy, angle = ment.cov.calc_rms_ellipse_params(proj.cov())
        angle = -math.degrees(angle)
        center = (0.0, 0.0)
        cx *= 4.0
        cy *= 4.0
        ax.add_patch(
            Ellipse(center, cx, cy, angle=angle, color=color, fill=False, ls=ls)
        )
plt.savefig(os.path.join(output_dir, "fig_results.png"), dpi=300)
plt.close()
