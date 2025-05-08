"""Fit ND covariance matrix to random 1D projections."""
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

cov_true = np.cov(x_true.T)
print(cov_true)

rng = np.random.default_rng(args.seed)


# Forward model
# --------------------------------------------------------------------------------------

class ProjectionTransform:
    def __init__(self, direction: np.ndarray) -> None:
        self.direction = direction

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x * self.direction, axis=1)[:, None]


transforms = []
directions = rng.normal(size=(args.nmeas, args.ndim))
for direction in directions:
    direction = np.random.normal(size=ndim)
    direction = direction / np.linalg.norm(direction)
    transform = ProjectionTransform(direction)
    transforms.append(transform)

bin_edges = np.linspace(-args.xmax, args.xmax, args.bins + 1)

diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(edges=bin_edges, axis=0)
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
    bound=1.00e+02,
    verbose=args.verbose,
)
cov_matrix, fit_result = fitter.fit(method=args.method, iters=args.iters)


# Print results
print(fit_result)
print(fitter.build_cov())


# Plot results
x = fitter.sample(100_000)
projections_pred = unravel(simulate(x, fitter.transforms, fitter.diagnostics))
projections_meas = unravel(fitter.projections)

ncols = min(args.nmeas, 10)
nrows = int(np.ceil(args.nmeas / ncols))

fig, axs = plt.subplots(
    ncols=ncols,
    nrows=nrows,
    figsize=(ncols * 1.1, nrows * 1.1), 
    sharey=True,
    sharex=True,
    constrained_layout=True
)
for i, ax in enumerate(axs.flat):
    values_pred = projections_pred[i].values
    values_meas = projections_meas[i].values
    ax.plot(values_pred / values_meas.max(), color="lightgray")
    ax.plot(values_meas / values_meas.max(), color="black", lw=0.0, marker=".", ms=2.0)
plt.savefig(os.path.join(output_dir, "fig_results.png"), dpi=300)
plt.close()





