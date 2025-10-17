import argparse
import os
import time

import numpy as np
import psdist as ps
import psdist.plot as psv
import scipy.interpolate
import ultraplot as plt
from tqdm import tqdm

import ment


plt.rc["cmap.discrete"] = False
plt.rc["cmap.sequential"] = "viridis"
plt.rc["figure.facecolor"] = "white"
plt.rc["grid"] = False


parser = argparse.ArgumentParser()
parser.add_argument("--ndim", type=int, default=4)
parser.add_argument("--nmodes", type=int, default=5)
parser.add_argument("--nsamp", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=1241)
parser.add_argument("--chains", type=int, default=1)
parser.add_argument("--burnin", type=int, default=1000)
parser.add_argument("--prop-scale", type=float, default=1.0)
parser.add_argument("--start-scale", type=float, default=1.0)
parser.add_argument("--plot", type=int, default=0)
args = parser.parse_args()


# Settings
ndim = args.ndim
size = 1_000_000
nmodes = args.nmodes
seed = args.seed

# Create gaussian particle distribution
rng = np.random.default_rng(seed)
mean = np.zeros(ndim)
cov = np.identity(ndim)
for i in range(ndim):
    for j in range(i):
        cov[i, j] = cov[j, i] = rng.uniform(-0.4, 0.4)
x = rng.multivariate_normal(mean, cov, size=size)

# Add gaussian blobs
for _ in range(nmodes):
    scale = rng.uniform(0.8, 1.5, size=ndim)
    loc = rng.uniform(-5.0, 3.0, size=ndim)
    x = np.vstack([x, rng.normal(loc=loc, scale=scale, size=(size // nmodes, ndim))])
x = x - np.mean(x, axis=0)

rng.shuffle(x)
x_true = np.copy(x)


# Compute ground-truth histogram. This will act as our ground-truth distribution function to sample from.
n_bins = 25
limits = ps.limits(x_true)
values, edges = np.histogramdd(x_true, bins=n_bins, range=limits, density=True)
hist = ps.Histogram(values=values, edges=edges)
print("hist.shape =", hist.shape)


# Interpolate to obtain a smooth density function.
prob_func = scipy.interpolate.RegularGridInterpolator(
    hist.coords,
    hist.values,
    method="linear",
    bounds_error=False,
    fill_value=0.0,
)


# Metropolis-Hastings
proposal_cov = np.eye(ndim) * args.prop_scale
start_loc = np.zeros(ndim)
start_cov = np.eye(ndim) * args.start_scale
start_point = np.random.multivariate_normal(start_loc, start_cov, size=args.chains)

sampler = ment.samp.MetropolisHastingsSampler(
    ndim=ndim,
    proposal_cov=proposal_cov,
    start=start_point,
    chains=args.chains,
    burnin=args.burnin,
    debug=True,
    shuffle=True,
    seed=args.seed,
)
x_samp = sampler(prob_func, size=args.nsamp)
assert x_samp.shape[0] == args.nsamp


# Plot results
if args.plot:
    grid = psv.CornerGrid(ndim, corner=True, figwidth=(1.25 * ndim))
    grid.set_limits(ps.limits(x_true, rms=2.5))
    grid.plot_hist(hist, cmap="mono")
    grid.plot(x_samp[:, :], kind="hist", alpha=0.0, diag_kws=dict(color="red"))
    grid.plot(
        x_samp[:1000, :],
        diag=False,
        kind="scatter",
        color="red",
        s=0.25,
    )
    plt.show()


