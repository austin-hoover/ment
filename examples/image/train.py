"""Test 2D MENT with high-resolution image."""
import argparse
import os
import pathlib
import pickle
import time
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import ment
import skimage
from tqdm import tqdm

from utils import gen_image
from utils import get_grid_points
from utils import rec_fbp
from utils import rec_sart
from utils import radon_transform

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["image.cmap"] = "Blues"
plt.rcParams["savefig.dpi"] = 300.0
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--im", type=str, default="tree", choices=["shepp", "leaf", "tree"])  # [to do] add other distributions
parser.add_argument("--im-blur", type=float, default=0.0)
parser.add_argument("--im-pad", type=int, default=0)
parser.add_argument("--im-res", type=int, default=256)
parser.add_argument("--nmeas", type=int, default=25)
parser.add_argument("--angle-max", type=float, default=180.0)
parser.add_argument("--angle-min", type=float, default=0.0)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.25)
parser.add_argument("--prior-scale", type=float, default=10.0)
parser.add_argument("--int-loop", type=int, default=0)
parser.add_argument("--show", action="store_true")
parser.add_argument("--verbose", type=int, default=0)

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

# Save args as pickled dictionary
filename = os.path.join(output_dir, "args.pkl")
with open(filename, "wb") as file:
    pickle.dump(vars(args), file)


# Ground truth image
# --------------------------------------------------------------------------------------

grid_res = args.im_res
grid_shape = (grid_res, grid_res)
grid_values = gen_image(args.im, res=args.im_res, blur=args.im_blur, pad=args.im_pad)
grid_values = grid_values.T
grid_values_true = grid_values.copy()

xmax = 1.0
grid_edges = 2 * [np.linspace(-xmax, xmax, grid_res + 1)]
grid_coords = [0.5 * (e[:-1] + e[1:]) for e in grid_edges]
grid_points = get_grid_points(grid_coords)

# Plot image
fig, ax = plt.subplots()
ax.pcolormesh(grid_coords[0], grid_coords[1], grid_values.T)
plt.savefig(os.path.join(output_dir, "fig_true_image.png"))
plt.close()


# Forward model
# --------------------------------------------------------------------------------------

angles = np.linspace(args.angle_min, args.angle_max, args.nmeas, endpoint=False)
angles = np.radians(angles)

transforms = []
for angle in angles:
    matrix = ment.utils.rotation_matrix(angle)
    transform = ment.sim.LinearTransform(matrix)
    transforms.append(transform)

diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(
        axis=0, 
        edges=grid_edges[0],
        thresh=0.0,
    )
    diagnostics.append([diagnostic])


# Training data
# --------------------------------------------------------------------------------------

print("Generating training data")

sinogram = radon_transform(grid_values_true, angles)

projections = []
for j in range(sinogram.shape[1]):
    projection = ment.Histogram1D(
        values=sinogram[:, j],
        coords=grid_coords[0],
        axis=0,
    )
    projection.normalize()
    projections.append([projection])


## The following method uses linear interpolation directly rather than calling
## `skimage.transform.radon`.
# import scipy.interpolate

# interp = scipy.interpolate.RegularGridInterpolator(
#     grid_coords, 
#     grid_values_true,
#     method="linear",
#     fill_value=0.0,
#     bounds_error=False,
# )

# projections = []
# for transform in tqdm(transforms):
#     grid_points_out = transform.inverse(grid_points)
#     grid_values_out = interp(grid_points_out)
#     grid_values_out = grid_values_out.reshape(grid_shape)
#     grid_values_out_proj = np.sum(grid_values_out, axis=1)

#     projection = ment.Histogram1D(
#         values=grid_values_out_proj,
#         coords=grid_coords[0],
#         axis=0,
#         thresh=0.1,
#         thresh_type="frac",
#     )
#     projections.append([projection])



# Plot sinogram
fig, ax = plt.subplots()
ax.pcolormesh(sinogram)
plt.savefig(os.path.join(output_dir, "fig_true_sinogram.png"))
plt.close()


# Reconstruction model
# --------------------------------------------------------------------------------------

# Define prior distribution for relative entropy
prior = ment.GaussianPrior(ndim=2, scale=args.prior_scale)

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


def plot_sinogram(model: ment.MENT):
    image_true = grid_values_true.copy()
    image_pred = model.prob(grid_points).reshape(grid_shape)

    sinogram_true = radon_transform(image_true, angles=angles)
    sinogram_pred = radon_transform(image_pred, angles=angles)

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
    values_true = grid_values_true.copy()
    values_pred = model.prob(grid_points).reshape(grid_shape)

    cell_volume = np.prod([c[1] - c[0] for c in grid_coords])
    values_true = values_true / np.sum(values_true) / cell_volume
    values_pred = values_pred / np.sum(values_pred) / cell_volume

    # Data discrepancy
    sinogram_true = radon_transform(values_true, angles=angles)
    sinogram_pred = radon_transform(values_pred, angles=angles)
    for j in range(sinogram_true.shape[1]):
        sinogram_true /= np.sum(sinogram_true) / np.sqrt(cell_volume)
        sinogram_pred /= np.sum(sinogram_pred) / np.sqrt(cell_volume)

    discrepancy = np.mean(np.abs(sinogram_pred - sinogram_true))

    # Absolute entropy    
    p = values_pred
    q = np.ones(p.shape)
    q = q / np.sum(q) / cell_volume
    entropy = -np.sum(p * np.log(1.00e-12 + p / q) * cell_volume)

    # Distance from true image
    distance_mae = np.mean(np.abs(values_pred - values_true))

    results = {}
    results["discrepancy"] = discrepancy
    results["distance_mae"] = distance_mae
    results["entropy"] = entropy
    return results


history = {}
for key in ["iteration", "discrepancy", "entropy", "distance_mae"]:
    history[key] = []


for iteration in range(args.iters):
    print("iteration =", iteration)

    # Update model
    if iteration > 0:
        model.gauss_seidel_step(
            learning_rate=args.lr,
            thresh=0.001,
            thresh_type="frac",
        )

    # Evaluate model
    results = evaluate_model(model)
    pprint(results)

    # Store history
    history["iteration"].append(iteration)
    for key in results:
        history[key].append(results[key])

    # Plot sinogram
    fig, axs = plot_sinogram(model)
    plt.savefig(os.path.join(output_dir, f"fig_proj_{iteration:02.0f}.png"))
    if args.show:
        plt.show()
    plt.close()

    # Plot image
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
plt.close()

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(history["iteration"], history["entropy"], marker=".")
ax.set_xlabel("Iteration")
ax.set_ylabel("Entropy")
plt.savefig(os.path.join(output_dir, f"fig_history_entropy.png"))
if args.show:
    plt.show()
plt.close()


# Other methods
# --------------------------------------------------------------------------------------

# Make dictionary for comparison:
results = {}
for method in ["fbp", "sart", "ment", "true"]:
    results[method] = {}
    for key in ["sinogram", "image"]:
        results[method][key] = None

# Form true image and sinogram
image_true = grid_values_true.copy()
projections = ment.unravel(model.projections)
sinogram_true = np.vstack([projection.values for projection in projections])
sinogram_true = sinogram_true.T

results["true"]["sinogram"] = sinogram_true.copy()
results["true"]["image"] = image_true.copy()


# FBP
image_pred = rec_fbp(sinogram_true, angles)
sinogram_pred = radon_transform(image_pred, angles=angles)
results["fbp"]["image"] = image_pred.copy()
results["fbp"]["sinogram"] = sinogram_pred.copy()

fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
for ax, image in zip(axs, [image_pred, image_true]):
    ax.pcolormesh(image.T)
plt.savefig(os.path.join(output_dir, "fig_other_fbp_image.png"))
plt.close()

fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
for ax, sinogram in zip(axs, [sinogram_pred, sinogram_true]):
    ax.pcolormesh(sinogram)
plt.savefig(os.path.join(output_dir, "fig_other_fbp_sinogram.png"))
plt.close()


# SART
image_pred = rec_sart(sinogram_true, angles, iterations=args.sart_iters)
sinogram_pred = radon_transform(image_pred, angles=angles)
results["sart"]["image"] = image_pred.copy()
results["sart"]["sinogram"] = sinogram_pred.copy()

fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
for ax, image in zip(axs, [image_pred, image_true]):
    ax.pcolormesh(image.T)
plt.savefig(os.path.join(output_dir, "fig_other_sart_image.png"))
plt.close()

fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
for ax, sinogram in zip(axs, [sinogram_pred, sinogram_true]):
    ax.pcolormesh(sinogram)
plt.savefig(os.path.join(output_dir, "fig_other_sart_sinogram.png"))
plt.close()


# MENT
image_pred = model.prob(grid_points).reshape(grid_shape)
sinogram_pred = radon_transform(image_pred, angles=angles)
results["ment"]["image"] = image_pred.copy()
results["ment"]["sinogram"] = sinogram_pred.copy()


# Compare
scale = 1.0 
for name in results:
    image = results[name]["image"]
    image = image / np.sum(image)
    results[name]["image"] = np.copy(image)
    scale = max(scale, np.max(image))
for name in results:
    results[name]["image"] /= scale

fig, axs = plt.subplots(ncols=4, figsize=(10, 2.5), sharex=True, sharey=True)
for ax, key in zip(axs, results):
    image = results[key]["image"]
    ax.pcolormesh(image.T, vmin=0.0, vmax=1.0)
    ax.set_title(key.upper())
plt.savefig(os.path.join(output_dir, "fig_compare_image.png"))
plt.close()

fig, axs = plt.subplots(ncols=4, figsize=(10, 2.5), sharex=True, sharey=True)
for ax, key in zip(axs, results):
    image = results[key]["sinogram"]
    ax.pcolormesh(image.T)
    ax.set_title(key.upper())
plt.savefig(os.path.join(output_dir, "fig_compare_sinogram.png"))
plt.close()

fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(10, 5.0))
for j, name in enumerate(results):
    for i, key in enumerate(["image", "sinogram"]):
        ax = axs[i, j]
        image = results[name][key]        
        ax.pcolormesh(image.T)
for j, name in enumerate(results):
    axs[0, j].set_title(name.upper())
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig(os.path.join(output_dir, "fig_compare_all.png"))
plt.close()