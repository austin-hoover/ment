"""ND reconstruction from random 1D projections."""
import argparse
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment
from ment.train.plot import Plotter
from ment.train.plot import PlotDistCorner
from ment.train.plot import PlotProj1D

plt.style.use("./style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dist",
    type=str,
    default="gaussian-mixture",
    choices=["gaussian-mixture", "rings", "gaussian", "waterbag", "kv"],
)
parser.add_argument("--ndim", type=int, default=6)
parser.add_argument("--nmeas", type=int, default=10)
parser.add_argument("--nbins", type=int, default=64)
parser.add_argument("--xmax", type=float, default=3.5)
parser.add_argument(
    "--mode", type=str, default="reverse", choices=["reverse", "forward"]
)
parser.add_argument("--samp-method", type=str, default="mh")
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.75)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
timestamp = time.strftime("%y%m%d_%H%M%S")
output_dir = os.path.join("outputs", path.stem, timestamp)
os.makedirs(output_dir, exist_ok=True)


# Source distribution
# --------------------------------------------------------------------------------------

ndim = args.ndim
xmax = args.xmax
seed = args.seed

dist = ment.dist.get_dist(args.dist, ndim=ndim, seed=seed)
x_true = dist.sample(1_000_000)
x_true = x_true.float()

limits = args.ndim * [(-xmax, xmax)]


# Data generation
# --------------------------------------------------------------------------------------


class ProjectionTransform:
    def __init__(self, direction: torch.Tensor) -> None:
        self.direction = direction

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        return torch.sum(x * self.direction, axis=1)[:, None]


transforms = []
for _ in range(args.nmeas):
    direction = torch.randn(ndim)
    direction = direction / torch.linalg.norm(direction)
    transform = ProjectionTransform(direction)
    transforms.append(transform)


# Create histogram diagnostic
axis_proj = axis_meas = 0
bin_edges = torch.linspace(-xmax, xmax, args.nbins + 1)

diagnostics = []
for transform in transforms:
    diagnostic = ment.Histogram1D(
        axis=axis_meas,
        edges=bin_edges,
    )
    diagnostics.append([diagnostic])


# Generate data from the source distribution.
projections = ment.simulate_with_diag_update(
    x_true,
    transforms,
    diagnostics,
    thresh=5.00e-03,
)


# Reconstruction model
# --------------------------------------------------------------------------------------

prior = ment.GaussianPrior(ndim=ndim, scale=1.0)

samp_method = args.samp_method

if samp_method == "grid":
    samp_grid_res = 32
    samp_noise = 0.5
    samp_grid_shape = ndim * [samp_grid_res]
    samp_grid_limits = limits

    sampler = ment.samp.GridSampler(
        grid_limits=samp_grid_limits,
        grid_shape=samp_grid_shape,
        noise=samp_noise,
    )

elif samp_method == "mh":
    samp_burnin = 500
    samp_chains = 1000
    samp_prop_cov = torch.eye(ndim) * (0.5**2)
    samp_start = torch.randn(samp_chains, ndim) * 0.5

    sampler = ment.MetropolisHastingsSampler(
        ndim=ndim,
        start=samp_start,
        proposal_cov=samp_prop_cov,
        burnin=samp_burnin,
        shuffle=True,
        verbose=1,
        noise=0.10,  # slight smoothing
        noise_type="gaussian",
    )

else:
    raise ValueError


model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    nsamp=100_000,
    mode="forward",
    verbose=True,
)


# Training
# --------------------------------------------------------------------------------------

plot_nsamp = x_true.shape[0]

plot_model = Plotter(
    n_samples=plot_nsamp,
    plot_proj=[
        PlotProj1D(log=False),
    ],
    plot_dist=[
        PlotDistCorner(
            fig_kws=dict(figsize=(ndim * 1.4, ndim * 1.4)),
            limits=(ndim * [(-xmax, xmax)]),
            bins=64,
        ),
    ],
)

eval_model = ment.train.Evaluator(nsamp=plot_nsamp)

trainer = ment.train.Trainer(
    model,
    plot_func=plot_model,
    eval_func=eval_model,
    output_dir=output_dir,
)

trainer.train(iters=3, lr=0.95)


# Evaluate
# --------------------------------------------------------------------------------------

x_pred = model.unnormalize(model.sample(1_000_000))

grid = ment.train.plot.CornerGrid(ndim, figsize=(ndim * 1.4, ndim * 1.4))
for i, x in enumerate([x_true, x_pred]):
    color = ["black", "red"][i]
    grid.plot(
        x,
        limits=limits,
        bins=64,
        proc_kws=dict(scale="max", blur=1.0),
        kind="contour",
        colors=color,
        diag_kws=dict(color=color, kind="line"),
        levels=np.linspace(0.01, 1.0, 7),
    )
plt.savefig(os.path.join(output_dir, "figures", "fig_corner_final"))
plt.close("all")
