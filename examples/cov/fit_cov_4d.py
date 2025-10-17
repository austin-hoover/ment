"""Fit 4D covariance matrix to 2D measurements."""
import argparse
import os
import pathlib
from typing import Callable
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import ment
from ment.diag import HistogramND
from ment.diag import Histogram1D
from ment.sim import simulate
from ment.utils import unravel
from ment.utils import rotation_matrix


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
print(np.cov(x_true.T))


# Forward model
# --------------------------------------------------------------------------------------

rng = np.random.default_rng(args.seed)
phase_advances = rng.uniform(0.0, np.pi, size=(args.nmeas, 2))
transfer_matrices = []
for mux, muy in phase_advances:
    matrix = np.eye(ndim)
    matrix[0:2, 0:2] = ment.sim.rotation_matrix(mux)
    matrix[2:4, 2:4] = ment.sim.rotation_matrix(muy)
    transfer_matrices.append(matrix)

transforms = []
for matrix in transfer_matrices:
    transform = ment.sim.LinearTransform(matrix)
    transforms.append(transform)

bin_edges = 2 * [np.linspace(-args.xmax, args.xmax, args.bins + 1)]

diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.HistogramND(axis=(0, 2), edges=bin_edges)
    diagnostics.append([diagnostic])


# Data
# --------------------------------------------------------------------------------------

projections = simulate(x_true, transforms, diagnostics)


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
def rms_ellipse_params(cov_matrix: np.ndarray) -> tuple[float, float, float]:
    sii = cov_matrix[0, 0]
    sjj = cov_matrix[1, 1]
    sij = cov_matrix[0, 1]

    angle = -0.5 * np.arctan2(2 * sij, sii - sjj)

    _sin = np.sin(angle)
    _cos = np.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2

    c1 = np.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = np.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))
    return (c1, c2, angle)


x = fitter.sample(100_000)
projections_pred = unravel(simulate(x, fitter.transforms, fitter.diagnostics))
projections_true = unravel(fitter.projections)

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

        cx, cy, angle = rms_ellipse_params(proj.cov())
        angle = -np.degrees(angle)
        center = (0.0, 0.0)
        cx *= 4.0
        cy *= 4.0
        ax.add_patch(
            Ellipse(center, cx, cy, angle=angle, color=color, fill=False, ls=ls)
        )
plt.savefig(os.path.join(output_dir, "fig_results.png"), dpi=300)
plt.close()
