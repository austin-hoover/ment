"""Simple 2D MENT reconstruction."""
import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import torch

import ment

plt.style.use("./style.mplstyle")


# Setup
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nmeas", type=int, default=7)
parser.add_argument("--mode", type=str, default="integrate")
parser.add_argument("--iters", type=int, default=4)
args = parser.parse_args()

ndim = 2
seed = 0

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Ground truth distribution
# --------------------------------------------------------------------------------------

rng = torch.Generator()
x_true = torch.randn(size=(1_000_000, ndim), generator=rng)
x_true = x_true / torch.linalg.norm(x_true, axis=1)[:, None]
x_true = x_true * 1.5
x_true = x_true + 0.25 * torch.randn(size=x_true.shape)


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
bin_edges = torch.linspace(-4.0, 4.0, 55)
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
prior = ment.GaussianPrior(ndim=2, scale=[1.0, 1.0])

# Define particle sampler (if mode="sample")
sampler = ment.GridSampler(
    limits=(2 * [(-4.0, 4.0)]),
    shape=(128, 128),
)

# Define integration grid (if mode="integrate"). We need separate integration
# limits for each measurement.
integration_limits = [(-4.0, 4.0)]
integration_limits = [[integration_limits for _ in diagnostics] for _ in transforms]
integration_size = 100

# Set up MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    integration_limits=integration_limits,
    integration_size=integration_size,
    integration_loop=True,
    mode=args.mode,
)


# Training
# --------------------------------------------------------------------------------------


def plot_model(model):
    # Sample particles
    x_pred = model.sample(100_000)

    # Plot simulated vs. measured profiles
    projections_true = ment.unravel(model.projections)
    projections_pred = ment.unravel(
        ment.simulate(x_pred, model.transforms, model.diagnostics)
    )

    fig, axs = plt.subplots(
        ncols=args.nmeas,
        figsize=(11.0, 1.0),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, ax in enumerate(axs):
        values_pred = projections_pred[i].values
        values_true = projections_true[i].values
        ax.plot(values_pred / values_true.max(), color="lightgray")
        ax.plot(
            values_true / values_true.max(), color="black", lw=0.0, marker=".", ms=2.0
        )
    return fig


for iteration in range(args.iters):
    print("iteration =", iteration)

    if iteration > 0:
        model.gauss_seidel_step(learning_rate=0.90)

    fig = plot_model(model)
    fig.savefig(os.path.join(output_dir, f"fig_proj_{iteration:02.0f}.png"))
    plt.close("all")


# Plot final distribution
x_pred = model.sample(x_true.shape[0])

fig, axs = plt.subplots(ncols=2, figsize=(6.0, 3.0), constrained_layout=True)
for ax, x in zip(axs, [x_pred, x_true]):
    ax.hist2d(x[:, 0], x[:, 1], bins=55, range=[(-4.0, 4.0), (-4.0, 4.0)])
fig.savefig(os.path.join(output_dir, f"fig_dist_{iteration:02.0f}.png"))
plt.close("all")
