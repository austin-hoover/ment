import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import ment


# Settings
ndim = 2
nmeas = 7
seed = 0

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Ground truth distribution
# --------------------------------------------------------------------------------------

rng = np.random.default_rng(seed)
x_true = rng.normal(size=(1_000_000, ndim))
x_true = x_true / np.linalg.norm(x_true, axis=1)[:, None]
x_true = x_true * 1.5
x_true = x_true + rng.normal(size=x_true.shape, scale=0.25)


# Forward model
# --------------------------------------------------------------------------------------

# Create a list of transforms. Each transform is a function with the call signature
# `transform(x: np.ndarray) -> np.ndarray`.


def rotation_matrix(angle: float) -> np.ndarray:
    M = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    M = np.array(M)
    return M


transforms = []
for angle in np.linspace(0.0, np.pi, nmeas, endpoint=False):
    M = rotation_matrix(angle)
    transform = ment.sim.LinearTransform(M)
    transforms.append(transform)

# Create a list of histogram diagnostics for each transform.
bin_edges = np.linspace(-4.0, 4.0, 55)
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
    grid_limits=(2 * [(-4.0, 4.0)]),
    grid_shape=(128, 128),
)

# Define integration grid (if mode="integrate"). You need separate integration
# limits for each measurement.
integration_limits = [
    (-4.0, 4.0),
]
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
    integration_loop=False,
    mode="integrate",
)


# Training
# --------------------------------------------------------------------------------------


def plot_model(model):
    # Sample particles
    x_pred = model.sample(100_000)

    # Plot sim vs. measured profiles
    projections_true = ment.unravel(model.projections)
    projections_pred = ment.unravel(
        ment.simulate(x_pred, model.transforms, model.diagnostics)
    )

    fig, axs = plt.subplots(
        ncols=nmeas,
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


for epoch in range(4):
    print("epoch =", epoch)

    if epoch > 0:
        model.gauss_seidel_step(learning_rate=0.90)

    fig = plot_model(model)
    fig.savefig(os.path.join(output_dir, f"fig_proj_{epoch:02.0f}.png"))
    plt.close("all")


# Plot final distribution
x_pred = model.sample(x_true.shape[0])

fig, axs = plt.subplots(ncols=2, figsize=(6.0, 3.0), constrained_layout=True)
for ax, x in zip(axs, [x_pred, x_true]):
    ax.hist2d(x[:, 0], x[:, 1], bins=55, range=[(-4.0, 4.0), (-4.0, 4.0)])
fig.savefig(os.path.join(output_dir, f"fig_dist_{epoch:02.0f}.png"))
plt.close("all")
