"""2D MENT reconstruction using particle sampling (forward mode)."""
import argparse
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment

plt.style.use("./style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dist",
    type=str,
    default="swissroll",
    choices=[
        "eight-gaussians",
        "galaxy",
        "gaussian-mixture",
        "hollow",
        "kv",
        "pinwheel",
        "rings",
        "swissroll",
        "two-spirals",
        "waterbag",
    ],
)
parser.add_argument("--nmeas", type=int, default=7)
parser.add_argument("--nbins", type=int, default=80)
parser.add_argument("--xmax", type=float, default=4.0)
parser.add_argument(
    "--samp-method",
    type=str,
    default="grid",
    choices=["grid", "mh", "nurs", "hmc", "flow"],
)
parser.add_argument("--samp-chains", type=int, default=50)
parser.add_argument("--samp-size", type=int, default=50_000)
parser.add_argument("--diag-blur", type=float, default=1.0)
parser.add_argument("--iters", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.75)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

ndim = 2
seed = 0

path = pathlib.Path(__file__)
timestamp = time.strftime("%y%m%d_%H%M%S")
output_dir = os.path.join("outputs", path.stem, timestamp)
os.makedirs(output_dir, exist_ok=True)


# Source distribution
# --------------------------------------------------------------------------------------

dist = ment.dist.get_dist(args.dist, ndim=ndim, seed=seed, normalize=True)
x_true = dist.sample(1_000_000)
x_true = x_true.float()


# Forward model
# --------------------------------------------------------------------------------------

# Create a list of transforms. Each transform is a function with the call signature
# `transform(x: np.ndarray) -> np.ndarray`.

transforms = []
for i in range(args.nmeas):
    angle = torch.pi * (i / args.nmeas)
    transform = ment.LinearTransform(ment.utils.rotation_matrix(angle))
    transforms.append(transform)

# Create a list of histogram diagnostics for each transform.
bin_edges = torch.linspace(-args.xmax, args.xmax, args.nbins)
diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])


# Training data
# --------------------------------------------------------------------------------------

# Here we simulate the projections; in real life the projections would be
# measured.
projections = ment.simulate(x_true, transforms, diagnostics)


# Reconstruction model
# --------------------------------------------------------------------------------------

# Define prior distribution for relative entropy calculation
prior = ment.GaussianPrior(ndim=2, scale=1.0)

# Define particle sampler
if args.samp_method == "grid":
    samp_grid_shape = (100, 100)
    samp_grid_limits = 2 * [(-args.xmax, args.xmax)]
    sampler = ment.samp.GridSampler(
        limits=samp_grid_limits,
        shape=samp_grid_shape,
        noise=0.0,
    )
if args.samp_method == "hmc":
    chains = args.samp_chains
    sampler = ment.HamiltonianMonteCarloSampler(
        ndim=ndim,
        start=torch.randn((chains, ndim)) * 0.25**2,
        step_size=0.25,
        steps_per_samp=10,
        burnin=10,
        verbose=1,
    )
if args.samp_method == "mh":
    chains = args.samp_chains
    sampler = ment.MetropolisHastingsSampler(
        ndim=ndim,
        start=torch.randn((chains, ndim)) * 0.25**2,
        proposal_cov=torch.eye(ndim) * 0.25**2,
        burnin=10,
        verbose=1,
    )
if args.samp_method == "nurs":
    chains = args.samp_chains
    sampler = ment.NURSSampler(
        ndim=ndim,
        start=torch.randn((chains, ndim)),
        step_size=1,
        max_doublings=10,
        threshold=1e-5,
    )


# Set up MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    mode="forward",
    nsamp=args.samp_size,
    diag_kws=dict(blur=args.diag_blur, thresh=0.001, thresh_type="frac"),
)


def plot_model(model: ment.MENT) -> list[plt.Figure]:
    figs = []

    # Sample particles
    x_pred = model.sample(100_000)

    # Simulate data
    projections_true = ment.unravel(model.projections)
    projections_pred = ment.unravel(
        ment.simulate(x_pred, model.transforms, model.diagnostics)
    )

    # Plot distribution
    limits = 2 * [(-args.xmax, args.xmax)]

    fig, axs = plt.subplots(ncols=2, figsize=(6.0, 3.0))
    for ax, x in zip(axs, [x_pred, x_true]):
        grid_values, grid_edges = np.histogramdd(
            x[: x_pred.shape[0]], bins=128, range=limits
        )
        ax.pcolormesh(grid_edges[0], grid_edges[1], grid_values.T)
    figs.append(fig)

    # Plot simulated vs. measured projections.
    ncols = min(args.nmeas, 7)
    nrows = int(np.ceil(args.nmeas / ncols))

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(1.90 * ncols, 1.25 * nrows),
        sharex=True,
        sharey=True,
    )
    for index in range(len(projections_true)):
        ax = axs.flat[index]
        proj_true = projections_true[index]
        proj_pred = projections_pred[index]
        scale = proj_true.values.max()
        ax.plot(proj_true.coords, proj_true.values / scale, color="lightgray")
        ax.plot(
            proj_pred.coords,
            proj_pred.values / scale,
            color="black",
            marker=".",
            lw=0,
            ms=1.0,
        )
        ax.set_ylim(ax.get_ylim()[0], 1.25)
        ax.set_xlim(limits[0])
    figs.append(fig)

    return figs


# Training loop
for iteration in range(args.iters):
    print("ITERATION =", iteration)

    if iteration > 0:
        model.gauss_seidel_step(lr=args.lr)

    for i, fig in enumerate(plot_model(model)):
        filename = f"fig_{i:02.0f}_{iteration:03.0f}"
        filename = os.path.join(output_dir, filename)
        fig.savefig(filename)
        if args.show:
            plt.show()
    plt.close("all")


print("Output directory:")
print(output_dir)
