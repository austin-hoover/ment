"""High-dynamic-range (HDR) tomography

The dynamic range is defined as the ratio of the maximum to minimum density in
the distribution; for example, a dynamic range of $10^3$ means the peak density
is one thousand times higher than the noise floor. Densities below $10^{-2}$
(as a fraction of the peak) are basically invisible to the naked eye and are
therefore unimportant in most applications. However, low-density regions can be
*very* important in high-power particle accelerators, where tiny fractional beam
loss contributes to intensity-limiting radioactivation in the accelerator.
"""

import math
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import ment

plt.style.use("../style.mplstyle")


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Helper functions
# --------------------------------------------------------------------------------------


def plot_image(
    values: torch.Tensor,
    coords: list[torch.Tensor],
    log: bool = True,
    colorbar: bool = False,
    ax=None,
    **plot_kws,
) -> None:
    """Plot image in log scale."""
    values = torch.as_tensor(values).clone()
    values = values + torch.min(values[values > 0.0])
    values = values / torch.max(values)
    if log:
        values = torch.log10(values + 1.00e-12)
        plot_kws.setdefault("vmax", 0.0)

    mesh = ax.pcolormesh(coords[0], coords[1], values.T, **plot_kws)
    if colorbar:
        fig.colorbar(mesh, ax=ax)


def plot_points(
    x: torch.Tensor, bins: int, limits: list[tuple[float, float]], **plot_kws
) -> None:
    """Plot histogram in log scale."""
    values, edges = np.histogramdd(x, bins=bins, range=limits)
    plot_image(values, edges, **plot_kws)


def sample_grid(
    values: np.ndarray, coords: list[np.ndarray], size: int, noise: float = 0.0
) -> np.ndarray:
    """Sample points from histogram."""
    pdf = np.ravel(values) / np.sum(values)
    idx = np.arange(pdf.size)
    idx = np.random.choice(idx, size, replace=True, p=pdf)
    idx = np.unravel_index(idx, shape=values.shape)

    edges = []
    for c in coords:
        dx = c[1] - c[0]
        e = np.zeros(len(c) + 1)
        e[0:-1] = c - 0.5 * dx
        e[-1] = c[-1] + 0.5 * dx
        edges.append(e)

    lb = [edges[axis][idx[axis]] for axis in range(values.ndim)]
    ub = [edges[axis][idx[axis] + 1] for axis in range(values.ndim)]

    x = np.zeros((size, values.ndim))
    for axis in range(x.shape[1]):
        x[:, axis] = np.random.uniform(lb[axis], ub[axis])
        if noise:
            delta = 0.5 * noise * (ub[axis] - lb[axis])
            x[:, axis] += np.random.uniform(-delta, delta, size=x.shape[0])
    return x


def norm_matrix_from_cov(cov_matrix: torch.Tensor) -> torch.Tensor:
    """Symplectic matrix that diagonalizes covariance matrix."""
    emittance = torch.sqrt(torch.linalg.det(cov_matrix))
    beta = cov_matrix[0, 0] / emittance
    alpha = -cov_matrix[0, 1] / emittance
    V = torch.tensor([[beta, 0.0], [-alpha, 1.0]]) * math.sqrt(1.0 / beta)
    A = torch.sqrt(torch.diag(torch.tensor([emittance, emittance])))
    V = torch.matmul(V, A)
    return torch.linalg.inv(V)


# Source distribution
# --------------------------------------------------------------------------------------

# As the source distribution, we use a simulated beam from the Beam Test Facility (BTF)
# at Oak Ridge National Laboratory.
grid = xr.open_dataarray("data/grid.nc")
grid_values = torch.as_tensor(grid.values)
grid_coords = [
    torch.as_tensor(grid.coords["x"].values),
    torch.as_tensor(grid.coords["xp"].values),
]
# Sample particles from the distribution and remove linear correlations.
x_true = sample_grid(
    values=grid.values, coords=grid.coords.values(), size=8_500_000, noise=1.0
)
x_true = torch.as_tensor(x_true).float()

cov_matrix = torch.cov(x_true.T)
norm_matrix = norm_matrix_from_cov(cov_matrix)
x_true = torch.matmul(x_true, norm_matrix.T)

xmax = 6.0
bins = 75
limits = 2 * [(-xmax, xmax)]


# Forward model
# --------------------------------------------------------------------------------------

# We consider a simple forward model consisting of rotation matrices by angles
# uniformly spaced between $[0, \pi]$.
nmeas = 10  # number of projections
nbins = 75  # number of bins in 1D histograms

# Define rotation angles uniformly spanning [0, pi].
phase_advances = torch.linspace(0.0, np.pi, nmeas + 1)[:-1]

# Package rotation matrices as LinearTransform function.
transforms = []
for phase_advance in phase_advances:
    matrix = ment.utils.rotation_matrix(phase_advance)
    transform = ment.LinearTransform(matrix)
    transforms.append(transform)

# Create Histogram1D objects.
diagnostics = []
for transform in transforms:
    bin_edges = torch.linspace(-xmax, xmax, nbins + 1)
    diagnostic = ment.diag.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])

# Generate training data from ground truth distribution.
projections = ment.simulate(x_true, transforms, diagnostics)


# Reconstruction model
# --------------------------------------------------------------------------------------

# Defina a wide Gaussian prior (basically uniform).
ndim = 2
prior = ment.GaussianPrior(ndim=ndim, scale=5.0)

# Create sampler to sample particles from the distribution (for plotting).
sampler = ment.GridSampler(limits=limits, shape=(200, 200))

# Define integration grid limits and resolution for each measurement. In this case
# the grid is one-dimenisonal.
integration_size = 100
integration_limits = [[limits] for transform in transforms]

# Create MENT reconstruction model.
model = ment.MENT(
    ndim=2,
    projections=projections,
    transforms=transforms,
    prior=prior,
    integration_size=integration_size,
    integration_limits=integration_limits,
    integration_loop=False,  # vectorized
    sampler=sampler,
    mode="integrate",
    verbose=False,
    diag_kws=dict(thresh=1e-6, thresh_type="frac"),
)


# Training
# --------------------------------------------------------------------------------------


def plot_model_proj(x_pred: np.ndarray) -> tuple:
    projections_pred = ment.simulate(x_pred, model.transforms, model.diagnostics)
    projections_pred = ment.unravel(projections_pred)
    projections_meas = ment.unravel(model.projections)

    fig, axs = plt.subplots(
        ncols=5, nrows=2, figsize=(10.5, 3.75), sharex=True, sharey=True
    )
    index = 0
    for index in range(nmeas):
        ax = axs.flat[index]
        projection_pred = projections_pred[index]
        projection_meas = projections_meas[index]
        scale = projection_meas.values.max()
        ax.plot(
            projection_pred.coords,
            torch.log10(projection_pred.values / scale + 1.00e-12),
            color="black",
            lw=1.5,
            alpha=0.25,
        )
        ax.plot(
            projection_meas.coords,
            torch.log10(projection_meas.values / scale + 1.00e-12),
            color="black",
            lw=0.0,
            ms=2.0,
            marker=".",
        )
        ax.set_ylim(-6.0, 0.5)
        ax.annotate(f"{index:02.0f}", xy=(0.02, 0.85), xycoords="axes fraction")
    for ax in axs.flat:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel("Log density")
    return fig, axs


def plot_model_dist(x_pred: np.ndarray) -> tuple:
    fig, axs = plt.subplots(ncols=2, figsize=(6.0, 3.0))
    for ax, x in zip(axs, [x_pred, x_true]):
        plot_points(x, bins=85, limits=limits, ax=ax, log=True)
    for ax in axs:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$p_x$")
    axs[0].set_title("MENT")
    axs[1].set_title("TRUE")
    return fig, axs


for iteration in range(7):
    print(f"iteration = {iteration}")

    # Sample particles from the distribution.
    x_pred = model.sample(x_true.shape[0])

    # Plot the phase space density.
    fig, axs = plot_model_dist(x_pred)
    filename = f"fig_dist_iter_{iteration:02.0f}"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close("all")

    # Plot the simulated vs. measured projections.
    plot_model_proj(x_pred)
    filename = f"fig_proj_iter_{iteration:02.0f}"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close("all")

    # Update the model parameters (Lagrange multipliers).
    model.gauss_seidel_step(learning_rate=0.85)
