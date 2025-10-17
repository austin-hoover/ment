import argparse
import os
import pathlib
import shutil
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import ment
import numpy as np
from scipy.ndimage import gaussian_filter

# local
from utils import make_dist
from utils import get_actions


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nsamp", type=int, default=500_000)
parser.add_argument("--nbins", type=int, default=90)
parser.add_argument("--blur", type=int, default=0.0)
parser.add_argument("--cmap", type=str, default="plasma")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


# Load model
# --------------------------------------------------------------------------------------

input_dir = "outputs/train/checkpoints"
filenames = os.listdir(input_dir)
filenames = sorted(filenames)
filenames = [f for f in filenames if f.endswith(".pt")]
filenames = [os.path.join(input_dir, f) for f in filenames]

filename = filenames[-1]

model = ment.MENT(
    ndim=4,
    transforms=None,
    projections=None,
    sampler=None,
    prior=None,
)
model.load(filename)


# Sample particles and simulate data
# --------------------------------------------------------------------------------------

nsamp = args.nsamp
x_true = make_dist(nsamp)
x_pred = model.unnormalize(model.sample(nsamp))
projections_pred = ment.unravel(
    ment.simulate(x_pred, model.transforms, model.diagnostics)
)
projections_true = ment.unravel(
    ment.simulate(x_true, model.transforms, model.diagnostics)
)


# Plot distribution of actions Jx-Jy
# --------------------------------------------------------------------------------------

ncols = len(projections_pred) // 2
cmap = args.cmap

for log in [False, True]:
    fig, axs = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(ncols * 2.0, 4.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for j, transform in enumerate(model.transforms):
        x_true_out = transform(x_true)
        x_pred_out = transform(x_pred)

        values_list = []
        for i, x_out in enumerate([x_pred_out, x_true_out]):
            actions = get_actions(x_out)
            sqrt_actions = np.sqrt(actions)

            xmax = 4.0
            limits = 2 * [(0.0, xmax)]
            values, edges = np.histogramdd(sqrt_actions, bins=args.nbins, range=limits)
            values_list.append(values)

        scale = np.max([np.max(values) for values in values_list])
        for i, values in enumerate(values_list):
            if args.blur:
                values = gaussian_filter(values, args.blur)
            # values = values / scale
            values = values / np.max(values)
            if log:
                values = np.log10(values + 1.00e-15)

            vmax = 1.0
            vmin = 0.0
            if log:
                vmax = 0.0
                vmin = -3.0

            ax = axs[i, j]
            mesh = ax.pcolormesh(
                edges[0],
                edges[1],
                values.T,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
                linewidth=0.0,
                rasterized=True,
                shading="auto",
            )

    axs[1, 0].set_xlabel("Jx")
    axs[1, 0].set_ylabel("Jy")

    filename = "fig_action"
    if log:
        filename = filename + "_log"
    filename = filename + ".pdf"
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300)
    plt.close()
