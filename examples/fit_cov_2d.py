import argparse
from typing import Callable
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

import ment
from ment.diag import HistogramND
from ment.diag import Histogram1D
from ment.sim import simulate
from ment.utils import unravel
from ment.utils import rotation_matrix


# Setup
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nmeas", type=int, default=6)
parser.add_argument("--xmax", type=float, default=7.0)
parser.add_argument("--bins", type=int, default=80)
parser.add_argument("--nsamp", type=int, default=1000)
args = parser.parse_args()

ndim = 2


# Ground truth distribution
# --------------------------------------------------------------------------------------

rng = np.random.default_rng(1234)
x_true = rng.normal(size=(1_000_000, ndim))
x_true = x_true / np.linalg.norm(x_true, axis=1)[:, None]
x_true = x_true * 1.5
x_true = x_true + rng.normal(size=x_true.shape, scale=0.25)
x_true = x_true / np.std(x_true, axis=0)
x_true[:, 0] *= 1.5
x_true[:, 1] /= 1.5
x_true = np.matmul(x_true, rotation_matrix(np.pi * 0.1).T)
print(np.cov(x_true.T))


# Forward model
# --------------------------------------------------------------------------------------

transforms = []
for angle in np.linspace(0.0, np.pi, args.nmeas, endpoint=False):
    M = rotation_matrix(angle)
    transform = ment.sim.LinearTransform(M)
    transforms.append(transform)

bin_edges = np.linspace(-args.xmax, args.xmax, args.bins)
diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])


# Data
# --------------------------------------------------------------------------------------

projections = simulate(x_true, transforms, diagnostics)


# Fit covariance matrix
# --------------------------------------------------------------------------------------
    
fitter = ment.CholeskyCovFitter(
    ndim=ndim, 
    transforms=transforms,
    projections=projections,
    nsamp=args.nsamp,
    bound=1.00e+06,
    verbose=True,
)
cov_matrix, fit_results = fitter.fit()


# Print results
print(cov_matrix)
print(fit_results)

# Plot results
x = fitter.sample(10_000)
projections_pred = unravel(simulate(x, fitter.transforms, fitter.diagnostics))
projections_meas = unravel(fitter.projections)

fig, axs = plt.subplots(
    ncols=args.nmeas, 
    figsize=(11.0, 1.0), 
    sharey=True,
    sharex=True,
    constrained_layout=True
)
for i, ax in enumerate(axs):
    values_pred = projections_pred[i].values
    values_meas = projections_meas[i].values
    ax.plot(values_pred / values_meas.max(), color="lightgray")
    ax.plot(values_meas / values_meas.max(), color="black", lw=0.0, marker=".", ms=2.0)
plt.show()




