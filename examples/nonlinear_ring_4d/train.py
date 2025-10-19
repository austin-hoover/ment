import argparse
import math
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import ment
from ment.sim import ComposedTransform
from ment.sim import LinearTransform

# Local
from lattice import AxiallySymmetricNonlinearKick
from utils import make_dist

plt.style.use("../style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dist-size", type=int, default=1_000_000)
parser.add_argument("--nmeas", type=int, default=4)
parser.add_argument("--nsamp", type=int, default=200_000)
parser.add_argument("--turn-step", type=int, default=20)
parser.add_argument("--diag-xmax", type=float, default=5.5)
parser.add_argument("--diag-bins", type=int, default=64)
parser.add_argument("--diag-blur", type=float, default=1.0)
parser.add_argument("--samp-xmax", type=float, default=3.0)
parser.add_argument("--samp-res", type=int, default=35)
parser.add_argument("--samp-noise", type=float, default=0.0)
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.90)
parser.add_argument("--plot-nsamp", type=int, default=None)
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

ndim = 4
rng = np.random.default_rng(args.seed)


# Initial distribution
# --------------------------------------------------------------------------------------

x_true = make_dist(args.dist_size, seed=args.seed)
cov_matrix = torch.cov(x_true.T)

cov_matrix = torch.cov(x_true.T)
V_inv = ment.cov.build_norm_matrix_from_cov(cov_matrix, scale=True)
V = torch.linalg.inv(V_inv)


# Forward model
# --------------------------------------------------------------------------------------

# Define lattice parameters
alpha = 0.0
beta = 1.0
phi = 2.0 * math.pi * 0.18

# Create transfer matrix
R = torch.eye(4)
R[0:2, 0:2] = ment.utils.rotation_matrix(phi)
R[2:4, 2:4] = ment.utils.rotation_matrix(phi)
M = torch.linalg.multi_dot([V, R, V_inv])

# Make lattice transform
lattice = ComposedTransform(
    LinearTransform(M),
    AxiallySymmetricNonlinearKick(alpha, beta, phi, A=0.75, E=0.5, T=1.0),
)

# Define phase space transformations
turns = list(range(0, args.turn_step * args.nmeas, args.turn_step))

transforms = []
for turn in turns:
    if turn == 0:
        transform = ment.IdentityTransform()
    else:
        transform = [lattice] * turn
        transform = ComposedTransform(*transform)
    transforms.append(transform)

# Create histogram diagnostics
xmax = args.diag_xmax
nbins = args.diag_bins

diag_grid_limits = 2 * [(-xmax, xmax)]
diag_grid_edges = [
    torch.linspace(-xmax, xmax, nbins + 1),
    torch.linspace(-xmax, xmax, nbins + 1),
]

diagnostics = []
for transform in transforms:
    diagnostic_x = ment.HistogramND(
        edges=diag_grid_edges, axis=(0, 1), blur=args.diag_blur
    )
    diagnostic_y = ment.HistogramND(
        edges=diag_grid_edges, axis=(2, 3), blur=args.diag_blur
    )
    diagnostics.append([diagnostic_x, diagnostic_y])


# Data generation
# --------------------------------------------------------------------------------------

projections = ment.simulate_with_diag_update(x_true, transforms, diagnostics, blur=0.0)


# Reconstruction model
# --------------------------------------------------------------------------------------

prior = ment.GaussianPrior(ndim=ndim, scale=1.0)

samp_grid_limits = ndim * [(-args.samp_xmax, args.samp_xmax)]
samp_grid_shape = ndim * [args.samp_res]
sampler = ment.samp.GridSampler(
    limits=samp_grid_limits, shape=samp_grid_shape, noise=args.samp_noise
)

model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    unnorm_matrix=V,
    mode="forward",
    nsamp=args.nsamp,
    verbose=True,
)


# Training
# --------------------------------------------------------------------------------------


def plot_func(model):
    figs = []

    nsamp = args.plot_nsamp or args.nsamp
    x_true = make_dist(nsamp)
    x_pred = model.unnormalize(model.sample(nsamp))
    projections_pred = ment.unravel(
        ment.simulate(x_pred, model.transforms, model.diagnostics)
    )
    projections_true = ment.unravel(
        ment.simulate(x_true, model.transforms, model.diagnostics)
    )

    # Plot x-y projections
    ncols = len(projections_pred)
    for log in [False, True]:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=ncols,
            figsize=(ncols * 1.75, 3.2),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        for j in range(ncols):
            projection_pred = projections_pred[j].copy()
            projection_true = projections_true[j].copy()
            scale = projection_true.values.max()
            for i, projection in enumerate([projection_true, projection_pred]):
                coords = projection.coords
                values = projection.values.clone()

                # values = values / scale
                values = values / values.max()
                if log:
                    values = torch.log10(values + 1.00e-15)

                vmax = 1.0
                vmin = 0.0
                if log:
                    vmax = +0.0
                    vmin = -3.0

                ax = axs[i, j]

                m = ax.pcolormesh(
                    coords[0],
                    coords[1],
                    gaussian_filter(values, 0.0).T,
                    cmap="plasma",
                    vmax=vmax,
                    vmin=vmin,
                    linewidth=0.0,
                    rasterized=True,
                    shading="auto",
                )
                if j == ncols - 1:
                    fig.colorbar(m, ax=ax)

        figs.append(fig)

    return figs


eval_model = ment.train.Evaluator(128_000)

trainer = ment.train.Trainer(
    model, eval_func=eval_model, plot_func=plot_func, output_dir=output_dir
)
trainer.train(iters=args.iters, lr=args.lr)
