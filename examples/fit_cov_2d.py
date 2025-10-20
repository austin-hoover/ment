"""Fit 2D covariance matrix to 1D measurements."""

import argparse
import math
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
parser.add_argument("--nmeas", type=int, default=6)
parser.add_argument("--xmax", type=float, default=7.0)
parser.add_argument("--bins", type=int, default=80)
parser.add_argument("--nsamp", type=int, default=1000)
parser.add_argument("--iters", type=int, default=1000)
parser.add_argument("--method", type=str, default="differential_evolution")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

ndim = 2


# Ground truth distribution
# --------------------------------------------------------------------------------------

x_true = torch.randn((1_000_000, ndim))
x_true = x_true / torch.linalg.norm(x_true, axis=1)[:, None]
x_true = x_true * 1.5
x_true = x_true + torch.randn(x_true.shape) * 0.25
x_true = x_true / torch.std(x_true, axis=0)
x_true[:, 0] *= 1.5
x_true[:, 1] /= 1.5
x_true = torch.matmul(x_true, ment.utils.rotation_matrix(math.pi * 0.1).T)

cov_matrix = torch.cov(x_true.T)
print(cov_matrix)


# Forward model
# --------------------------------------------------------------------------------------

transforms = []
for angle in torch.linspace(0.0, np.pi, args.nmeas + 1)[:-1]:
    M = ment.utils.rotation_matrix(angle)
    transform = ment.sim.LinearTransform(M)
    transforms.append(transform)

bin_edges = torch.linspace(-args.xmax, args.xmax, args.bins)
diagnostics = []
for transform in transforms:
    diagnostic = ment.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])


# Data
# --------------------------------------------------------------------------------------

projections = ment.simulate(x_true, transforms, diagnostics)


# Fit covariance matrix
# --------------------------------------------------------------------------------------

fitter = ment.CholeskyCovFitter(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    nsamp=args.nsamp,
    bound=1.00e06,
    verbose=True,
)
cov_matrix, fit_results = fitter.fit(iters=args.iters, method=args.method)


# Print results
print(cov_matrix)
print(fit_results)

# Plot results
x = fitter.sample(100_000)
projections_pred = ment.unravel(ment.simulate(x, fitter.transforms, fitter.diagnostics))
projections_meas = ment.unravel(fitter.projections)

fig, axs = plt.subplots(
    ncols=args.nmeas,
    figsize=(11.0, 1.0),
    sharey=True,
    sharex=True,
)
for i, ax in enumerate(axs):
    values_pred = projections_pred[i].values
    values_meas = projections_meas[i].values
    ax.plot(values_pred / values_meas.max(), color="lightgray")
    ax.plot(values_meas / values_meas.max(), color="black", lw=0.0, marker=".", ms=2.0)
plt.savefig(os.path.join(output_dir, "fig_results.png"), dpi=300)
plt.close()
