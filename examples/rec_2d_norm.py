"""2D MENT reconstruction in normalized coordinates."""
import argparse
import math
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
    "--mode", type=str, default="reverse", choices=["reverse", "forward"]
)
parser.add_argument("--iters", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.90)
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

# Add linear transformation
M = torch.eye(ndim)
M[0, 0] = 0.35
R = ment.utils.rotation_matrix(math.pi * 0.25)
M = torch.matmul(R, M)

x_true = torch.matmul(x_true, M.T)


# Forward model
# --------------------------------------------------------------------------------------

# Create a list of transforms.
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

# Get normalization matrix from covariance
cov_matrix = torch.cov(x_true.T)
norm_matrix = ment.cov.build_norm_matrix_from_cov(cov_matrix, scale=True)
unnorm_matrix = torch.linalg.inv(norm_matrix)

print("V:")
print(unnorm_matrix)

# Define prior distribution for relative entropy calculation
prior = ment.GaussianPrior(ndim=2, scale=1.0)

# Define particle sampler
samp_grid_shape = (100, 100)
samp_grid_limits = 2 * [(-args.xmax, args.xmax)]
sampler = ment.samp.GridSampler(
    limits=samp_grid_limits,
    shape=samp_grid_shape,
    noise=0.0,
)

# Define integration grid. We need separate integration limits for each measurement.
integration_limits = [(-args.xmax, args.xmax)]
integration_limits = [integration_limits for transform in transforms]
integration_size = 200

# Set up MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    unnorm_matrix=unnorm_matrix,
    sampler=sampler,
    integration_limits=integration_limits,
    integration_size=integration_size,
    integration_loop=False,
    mode=args.mode,
)


def plot_model(model: ment.MENT) -> list[plt.Figure]:
    figs = []

    # Sample particles
    x_pred = model.unnormalize(model.sample(1_000_000))

    # Simulate data
    projections_true = ment.unravel(model.projections)
    projections_pred = ment.unravel(
        ment.simulate(x_pred, model.transforms, model.diagnostics)
    )

    # Plot distribution
    limits = 2 * [(-args.xmax, args.xmax)]

    fig, axs = plt.subplots(ncols=2, figsize=(6.0, 3.0))
    for i, ax in enumerate(axs):
        grid_values, grid_edges = np.histogramdd(x_pred, bins=100, range=limits)
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
