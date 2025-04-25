"""Test 2D MENT with high-resolution image."""
import argparse
import os
import pathlib
import time
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import ment
import scipy.interpolate
import skimage
from tqdm import tqdm

from utils import gen_image
from utils import get_grid_points
from utils import rec_fbp
from utils import rec_sart

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["image.cmap"] = "Blues"
plt.rcParams["savefig.dpi"] = 300.0
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dist", type=str, default="shepp", choices=["shepp", "leaf", "tree"])
parser.add_argument("--res", type=int, default=256)
parser.add_argument("--nmeas", type=int, default=50)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.10)
parser.add_argument("--int-loop", type=int, default=0)
parser.add_argument("--show", action="store_true")
parser.add_argument("--verbose", type=int, default=2)

parser.add_argument("--sart-iters", type=int, default=5)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

ndim = 2
nmeas = args.nmeas

path = pathlib.Path(__file__)
timestamp = time.strftime("%y%m%d%H%M%S")
output_dir = os.path.join("outputs", path.stem, timestamp)
os.makedirs(output_dir, exist_ok=True)


# Ground truth distribution
# --------------------------------------------------------------------------------------

grid_res = args.res
grid_shape = (grid_res, grid_res)
grid_values = gen_image(args.dist, res=args.res)
grid_values = grid_values.T

xmax = 1.0
grid_edges = 2 * [np.linspace(-xmax, xmax, grid_res + 1)]
grid_coords = [0.5 * (e[:-1] + e[1:]) for e in grid_edges]
grid_points = get_grid_points(grid_coords)
grid_values_true = grid_values.copy()

# Plot image
fig, ax = plt.subplots()
ax.pcolormesh(grid_coords[0], grid_coords[1], grid_values.T)
plt.savefig(os.path.join(output_dir, "fig_true_image.png"))
plt.close()


# Forward model
# --------------------------------------------------------------------------------------

def rotation_matrix(angle: float) -> np.ndarray:
    M = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    M = np.array(M)
    return M

angles = np.linspace(0.0, np.pi, nmeas, endpoint=False)

transforms = []
for angle in angles:
    M = rotation_matrix(angle)
    transform = ment.sim.LinearTransform(M)
    transforms.append(transform)

bin_edges = grid_edges[0]
diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(
        axis=0, edges=bin_edges, thresh=0.01, thresh_type="frac"
    )
    diagnostics.append([diagnostic])


# Training data
# --------------------------------------------------------------------------------------

print("Generating training data")

sinogram = skimage.transform.radon(grid_values_true.T, theta=-np.degrees(angles))

projections = []
for j in range(sinogram.shape[1]):
    projection = ment.Histogram1D(
        values=sinogram[:, j],
        coords=grid_coords[0],
        axis=0,
    )
    projections.append([projection])

# Plot sinogram
fig, ax = plt.subplots()
ax.pcolormesh(sinogram)
plt.savefig(os.path.join(output_dir, "fig_true_sinogram.png"))
plt.close()


# Reconstruction model
# --------------------------------------------------------------------------------------

# Define prior distribution for relative entropy
prior = ment.GaussianPrior(ndim=2, scale=10.0)

# Define integration grid for each projection.
integration_limits = [[(-xmax, xmax)] for _ in transforms]
integration_size = grid_shape[0]
integration_loop = bool(args.int_loop)

# Create MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=None,
    integration_limits=integration_limits,
    integration_size=integration_size,
    integration_loop=integration_loop,
    mode="integrate",
    verbose=args.verbose,
)


# Training
# --------------------------------------------------------------------------------------

def collect_sinograms(model: ment.MENT) -> list[np.ndarray]:
    projections_true = ment.unravel(model.projections)
    projections_pred = ment.unravel(model.simulate())

    sinogram_true = np.zeros((len(projections_true), grid_res))
    for i in range(len(projections_true)):
        sinogram_true[i, :] = projections_true[i].values

    sinogram_pred = np.zeros((len(projections_pred), grid_res))
    for i in range(len(projections_pred)):
        sinogram_pred[i, :] = projections_pred[i].values

    return (sinogram_pred, sinogram_true)

    
def plot_projections(model: ment.MENT):
    (sinogram_pred, sinogram_true) = collect_sinograms(model)
    
    fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
    for ax, sinogram in zip(axs, [sinogram_pred, sinogram_true]):
        ax.pcolormesh(sinogram.T)
    return fig, axs


def plot_image(model: ment.MENT):
    image_true = grid_values_true.copy()
    image_pred = model.prob(grid_points).reshape(grid_shape)
    fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
    for ax, image in zip(axs, [image_pred, image_true]):
        ax.pcolormesh(image.T)
    return fig, axs


def evaluate_model(model: ment.MENT) -> dict:
    (sinogram_pred, sinogram_true) = collect_sinograms(model)
    
    discrepancy = np.mean(np.abs(sinogram_pred - sinogram_true))

    entropy = None
    
    results = {}
    results["discrepancy"] = discrepancy
    results["entropy"] = entropy    
    return results


history = {}
for key in ["iteration", "discrepancy", "entropy"]:
    history[key] = []
    
for iteration in range(args.iters):
    print("iteration =", iteration)

    if iteration > 0:
        model.gauss_seidel_step(learning_rate=args.lr)

    results = evaluate_model(model)
    pprint(results)

    history["iteration"].append(iteration)
    for key in results:
        history[key].append(results[key])

    fig, axs = plot_projections(model)
    plt.savefig(os.path.join(output_dir, f"fig_proj_{iteration:02.0f}.png"))
    if args.show:
        plt.show()
    plt.close()

    fig, axs = plot_image(model)
    plt.savefig(os.path.join(output_dir, f"fig_dist_{iteration:02.0f}.png"))
    if args.show:
        plt.show()
    plt.close()


# Plot training history
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(history["iteration"], history["discrepancy"], marker=".")
ax.set_xlabel("Iteration")
ax.set_ylabel("Discrepancy")
plt.savefig(os.path.join(output_dir, f"fig_history_discrepancy.png"))
if args.show:
    plt.show()


# Other methods
# --------------------------------------------------------------------------------------

# Form sinogram
projections = ment.unravel(model.projections)
sinogram = np.vstack([projection.values for projection in projections])


# FBP
image_pred = rec_fbp(sinogram, np.degrees(angles), iterations=5)
image_true = grid_values_true.copy()

fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
for ax, image in zip(axs, [image_pred, image_true]):
    ax.pcolormesh(image.T)
plt.savefig(os.path.join(output_dir, "fig_fbp_image.png"))
plt.close()


# SART
image_pred = rec_sart(sinogram, np.degrees(angles), iterations=5)
image_true = grid_values_true.copy()

fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
for ax, image in zip(axs, [image_pred, image_true]):
    ax.pcolormesh(image.T)
plt.savefig(os.path.join(output_dir, "fig_sart_image.png"))
plt.close()

