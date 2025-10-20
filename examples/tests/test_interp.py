"""Test interpolation."""
import argparse
import os
import pathlib

import torch
import matplotlib.pyplot as plt

import ment

plt.style.use("../style.mplstyle")


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# 1D
# --------------------------------------------------------------------------------------

xmax = 5.0
bins = 25

coords = torch.linspace(-xmax, xmax, bins)
values = torch.exp(-(coords**2)) + 0.5
lfunc = ment.LagrangeFunction(ndim=1, axis=0, coords=coords, values=values)

x = torch.zeros((bins**2, 2))
x[:, 0] = torch.linspace(-2.0 * xmax, 2.0 * xmax, x.shape[0])

f = lfunc(x)

fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(x[:, 0], f)
ax.plot(coords, values, marker=".", lw=0)
ax.set_ylim(-0.1, ax.get_ylim()[1])
plt.savefig(os.path.join(output_dir, "fig_interp_1d"))


# 2D
# --------------------------------------------------------------------------------------

xmax = 5.0
bins = 20

grid_edges = 2 * [torch.linspace(-xmax, xmax, bins + 1)]
grid_coords = [ment.utils.edges_to_coords(e) for e in grid_edges]
grid_values = torch.rand((bins, bins)) ** 2
lfunc = ment.LagrangeFunction(
    ndim=2, axis=(0, 1), coords=grid_coords, values=grid_values
)

new_grid_res = int(bins * 2)
new_grid_edges = 2 * [torch.linspace(-xmax, xmax, new_grid_res + 1)]
new_grid_coords = [ment.utils.edges_to_coords(e) for e in new_grid_edges]
x = torch.stack(
    [c.ravel() for c in torch.meshgrid(*new_grid_coords, indexing="ij")], axis=-1
)
f = lfunc(x)

new_grid_values = f.reshape((new_grid_res, new_grid_res))

old_grid_coords = grid_coords
old_grid_values = grid_values

fig, axs = plt.subplots(figsize=(6, 3), ncols=2, sharex=True, sharey=True)
axs[0].pcolormesh(old_grid_coords[0], old_grid_coords[1], old_grid_values.T, vmin=0.0)
axs[1].pcolormesh(new_grid_coords[0], new_grid_coords[1], new_grid_values.T, vmin=0.0)
for ax in axs:
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)
plt.savefig(os.path.join(output_dir, "fig_interp_2d"))
