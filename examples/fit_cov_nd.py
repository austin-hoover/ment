"""Fit ND covariance matrix to random 1D projections."""

import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment

plt.style.use("./style.mplstyle")


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--ndim", type=int, default=6)
parser.add_argument("--dist", type=str, default="gaussian-mixture")
parser.add_argument("--nmeas", type=int, default=30)
parser.add_argument("--xmax", type=float, default=4.0)
parser.add_argument("--bins", type=int, default=80)
parser.add_argument("--nsamp", type=int, default=1000)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--iters", type=int, default=500)
parser.add_argument("--method", type=str, default="differential_evolution")
parser.add_argument("--pop", type=int, default=5)
parser.add_argument("--verbose", type=int, default=2)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Source distribution
# --------------------------------------------------------------------------------------

ndim = args.ndim
dist = ment.dist.get_dist(args.dist, ndim=ndim, seed=args.seed)
x_true = dist.sample(1_000_000)

cov_true = torch.cov(x_true.T)
print(cov_true)


# Forward model
# --------------------------------------------------------------------------------------


class ProjectionTransform:
    def __init__(self, direction: torch.Tensor) -> None:
        self.direction = direction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.direction, axis=1)[:, None]


directions = torch.randn(args.nmeas, args.ndim)
directions = directions / torch.linalg.norm(directions, axis=1)[:, None]

transforms = []
for direction in directions:
    transform = ProjectionTransform(direction)
    transforms.append(transform)

bin_edges = torch.linspace(-args.xmax, args.xmax, args.bins + 1)

diagnostics = []
for transform in transforms:
    diagnostic = ment.Histogram1D(edges=bin_edges, axis=0)
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
    verbose=args.verbose,
)
cov_matrix, fit_result = fitter.fit(method=args.method, iters=args.iters)


# Print results
print(fit_result)
print(fitter.build_cov())


# Plot results
x = fitter.sample(100_000)
projections_pred = ment.unravel(ment.simulate(x, fitter.transforms, fitter.diagnostics))
projections_meas = ment.unravel(fitter.projections)

ncols = min(args.nmeas, 10)
nrows = int(np.ceil(args.nmeas / ncols))

fig, axs = plt.subplots(
    ncols=ncols,
    nrows=nrows,
    figsize=(ncols * 1.1, nrows * 1.1),
    sharey=True,
    sharex=True,
    constrained_layout=True,
)
for i, ax in enumerate(axs.flat):
    values_pred = projections_pred[i].values
    values_meas = projections_meas[i].values
    ax.plot(values_pred / values_meas.max(), color="lightgray")
    ax.plot(values_meas / values_meas.max(), color="black", lw=0.0, marker=".", ms=2.0)
plt.savefig(os.path.join(output_dir, "fig_results.png"), dpi=300)
plt.close()
