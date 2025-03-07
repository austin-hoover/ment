"""Plot 4D reconstruction results."""
import argparse
import os
import pathlib

import numpy as np
import ment
import matplotlib.pyplot as plt

# local
from tools.utils import list_paths

plt.style.use("tools/style.mplstyle")


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nsamp", type=int, default=1_000_000)
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

x_pred = model.unnormalize(model.sample(args.nsamp))


# Simulate data
# --------------------------------------------------------------------------------------

projections_meas = model.projections
projections_pred = ment.simulate(x_pred, model.transforms, model.diagnostics)


# Plot measured vs. simulated projections.
# --------------------------------------------------------------------------------------

fig, axs = plt.subplots(
    nrows=6,
    ncols=6,
    figsize=(6.0, 4.5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

index = 0
for i in range(len(projections_meas)):
    for j in range(len(projections_meas[i])):
        proj_meas = projections_meas[i][j].copy()
        proj_pred = projections_pred[i][j].copy()
        scale = np.max(proj_meas.values)

        ax = axs.flat[index]
        ax.plot(
            proj_pred.coords, 
            proj_pred.values / scale, 
            label="pred",
            color="red", 
            alpha=0.3,
        )
        ax.plot(
            proj_meas.coords, 
            proj_meas.values / scale, 
            label="meas",
            color="black",
            lw=0,
            marker=".",
            ms=1.0,
        )
        ax.annotate(
            "{:02.0f}".format(index // 3),
            xy=(0.03, 0.96),
            xycoords="axes fraction",
            horizontalalignment="left",
            verticalalignment="top",
        )
        index += 1

for ax in axs.flat:
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([])
    ax.set_xticks([-35.0, 0.0, 35.0])

axs[-1, 0].set_xlabel(r"$x$ (mm)")
axs[-1, 1].set_xlabel(r"$y$ (mm)")
axs[-1, 2].set_xlabel(r"$u$ (mm)")
axs[-1, 3].set_xlabel(r"$x$ (mm)")
axs[-1, 4].set_xlabel(r"$y$ (mm)")
axs[-1, 5].set_xlabel(r"$u$ (mm)")

filename = "fig_profiles.pdf"
filename = os.path.join(output_dir, filename)
plt.savefig(filename, dpi=250)
plt.close()


# Plot 2D projections of 4D distribution
# --------------------------------------------------------------------------------------

axes_proj = [(0, 1), (2, 3), (0, 2), (0, 3), (2, 1), (1, 3)]
dims = [r"$x$", "$x'$", "$y$", "$y'$"]
units = ["mm", "mrad", "mm", "mrad"]

xmax = 3.5 * np.std(x_pred, axis=0)
limits = list(zip(-xmax, xmax))

fig, axs = plt.subplots(
    ncols=3,
    nrows=2,
    figsize=(7.0, 4.0),
    sharex=False,
    sharey=False,
    constrained_layout=True,
)
for j, axis in enumerate(axes_proj):
    values, edges = np.histogramdd(
        x_pred[:, axis], 
        bins=64, 
        range=[limits[k] for k in axis], 
        density=True,
    )
    
    ax = axs.flat[j]
    ax.pcolormesh(
        edges[0],
        edges[1],
        values.T,
        rasterized=True,
        edgecolor="None",
        linewidth=0.0,
    )

for j, axis in enumerate(axes_proj):
    ax = axs.flat[j]
    ax.set_xlabel(f"{dims[axis[0]]} ({units[axis[0]]})")
    ax.set_ylabel(f"{dims[axis[1]]} ({units[axis[1]]})")

filename = f"fig_proj2d.pdf"
filename = os.path.join(output_dir, filename)
plt.savefig(filename, dpi=250)
plt.close("all")
