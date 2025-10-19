import argparse
import math
import os
import pathlib

import torch
import matplotlib.pyplot as plt

import ment

plt.style.use("./style.mplstyle")


# Setup
path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, default=64)
parser.add_argument("--n", type=int, default=25_000)
parser.add_argument("--blur", type=float, default=0.0)
parser.add_argument("--thresh", type=float, default=0.0)
parser.add_argument("--thresh-type", type=str, default="abs")
args = parser.parse_args()

# Make distribution
x = torch.randn((args.n, 2))
x = x / torch.linalg.norm(x, axis=1)[:, None]
x = x + torch.randn((args.n, 2)) * 0.25
x = x / torch.std(x, axis=0)

# Test 1D histogram
grid_coords = torch.linspace(-5.0, 5.0, args.bins)
diag = ment.Histogram1D(
    axis=0,
    coords=grid_coords,
    blur=args.blur,
    thresh=args.thresh,
    thresh_type=args.thresh_type,
)
grid_values = diag(x)

fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(grid_coords, grid_values)
plt.savefig(os.path.join(output_dir, "fig_hist_1d"))


# Test 2D histogram
grid_coords = 2 * [torch.linspace(-5.0, 5.0, args.bins)]
diag = ment.HistogramND(
    axis=(0, 1),
    coords=grid_coords,
    blur=args.blur,
    thresh=args.thresh,
    thresh_type=args.thresh_type,
)
grid_values = diag(x)

fig, ax = plt.subplots(figsize=(3, 3))
ax.pcolormesh(grid_coords[0], grid_coords[1], grid_values.T)
plt.savefig(os.path.join(output_dir, "fig_hist_2d"))
