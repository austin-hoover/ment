"""Plot 4D reconstruction in action space."""

import argparse
import os
import pathlib

import numpy as np
import ment
import matplotlib.pyplot as plt
import scipy.ndimage

from tools.cov import normalization_matrix
from tools.utils import list_paths


plt.style.use("tools/style.mplstyle")


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nsamp", type=int, default=1_000_000)
parser.add_argument("--nbins", type=int, default=64)
parser.add_argument("--jmax", type=float, default=85.0)
parser.add_argument("--contours", type=int, default=8)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

input_dir = "./outputs/train/"

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load model
# --------------------------------------------------------------------------------------

checkpoints_folder = os.path.join(input_dir, "checkpoints")
checkpoint_filenames = list_paths(os.path.join(input_dir, "checkpoints"), sort=True)
checkpoint_filename = checkpoint_filenames[-1]

model = ment.MENT(
    ndim=4,
    transforms=None,
    projections=None,
    prior=None,
    sampler=None,
)
model.load(checkpoint_filenames[-1])
model.sampler.noise = 1.0


# Sample particles from distribution
# --------------------------------------------------------------------------------------

x = model.unnormalize(model.sample(args.nsamp))

cov_matrix = np.cov(x.T)
norm_matrix = normalization_matrix(cov_matrix, scale=False)

z = np.matmul(x, norm_matrix.T)
cov_matrix_n = np.cov(z.T)

print("covariance matrix:")
print(np.round(cov_matrix, 5))

print("normalized covariance matrix:")
print(np.round(cov_matrix_n, 5))

eps1 = np.sqrt(cov_matrix_n[0, 0] * cov_matrix_n[1, 1])
eps2 = np.sqrt(cov_matrix_n[2, 2] * cov_matrix_n[3, 3])
eps_avg = np.sqrt(eps1 * eps2)  # maintains 4D emittance but equal mode amplitudes

z_pred = np.copy(z)


# Plot actions
# --------------------------------------------------------------------------------------


def make_joint_grid(figwidth=5.0, panel_width=0.33) -> tuple:
    fig, axs = plt.subplots(
        ncols=2,
        nrows=2,
        sharex="col",
        sharey="row",
        figsize=(figwidth, figwidth),
        gridspec_kw=dict(
            width_ratios=[1.0, panel_width],
            height_ratios=[panel_width, 1.0],
        ),
    )
    axs[0, 1].axis("off")
    return fig, axs


def plot_hist(
    values: np.ndarray, edges: list[np.ndarray], contours: int = 0, **plot_kws
) -> tuple:
    fig, axs = make_joint_grid()

    ax = axs[1, 0]
    ax.pcolormesh(
        edges[0],
        edges[1],
        values.T,
        linewidth=0.0,
        rasterized=True,
        shading="auto",
    )

    if contours:
        coords = [0.5 * (e[:-1] + e[1:]) for e in edges]
        axs[1, 0].contour(
            coords[0],
            coords[1],
            scipy.ndimage.gaussian_filter(values, 1.0).T,
            levels=np.linspace(0.01, 1.0, contours),
            colors="white",
            linewidths=0.80,
            alpha=0.15,
        )

    ax = axs[0, 0]
    proj_edges = edges[0]
    proj_values = np.sum(values, axis=1)
    proj_values = proj_values / np.max(proj_values)
    ax.stairs(proj_values, proj_edges, lw=1.5, color="black")
    ax.set_ylim(0.0, 1.25)

    ax = axs[1, 1]
    proj_edges = edges[1]
    proj_values = np.sum(values, axis=0)
    proj_values = proj_values / np.max(proj_values)
    ax.stairs(proj_values, proj_edges, orientation="horizontal", lw=1.5, color="black")
    ax.set_xlim(0.0, 1.25)

    return fig, axs


z = z_pred
j1 = np.sum(np.square(z[:, (0, 1)]), axis=1)
j2 = np.sum(np.square(z[:, (2, 3)]), axis=1)

edges = np.linspace(0.0, args.jmax, args.nbins + 1)
edges = [edges, edges]
values, _, _ = np.histogram2d(j1, j2, bins=edges)
values = values / np.max(values)

fig, axs = plot_hist(values, edges, contours=args.contours)
axs[1, 0].set_xlabel(r"$J_1$")
axs[1, 0].set_ylabel(r"$J_2$")

filename = f"fig_action.pdf"
filename = os.path.join(output_dir, filename)
plt.savefig(filename)
