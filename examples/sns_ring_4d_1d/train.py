"""Reconstruct 4D phase space distribution from 1D projections.

Usage:
    python train.py --samp-res=40
"""
import argparse
import os
import pathlib
import shutil
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import ment
import ment.train

from tools.cov import compute_cov_xy
from tools.cov import fit_cov
from tools.cov import normalization_matrix
from tools.profile import Profile
from tools.utils import list_paths


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.20)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--samp", type=int, default=1_000_000)
parser.add_argument("--samp-noise", type=float, default=0.25)
parser.add_argument("--samp-xmax", type=float, default=4.0)
parser.add_argument("--samp-res", type=int, default=32)
args = parser.parse_args()


# Setup
# -----------------------------------------------------------------------------------

input_dirs = {}
input_dirs["optics"] = "./outputs/00_save_optics"
input_dirs["data"] = "./outputs/01_proc_data"

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


# Load data
# --------------------------------------------------------------------------------------

tmats = [np.loadtxt(filename) for filename in list_paths(input_dirs["optics"])]
tmats = np.array(tmats)

profiles = {}
for dim in ["x", "y", "u"]:
    profiles[dim] = []
    for filename in list_paths(input_dirs["data"], startswith=f"profile_{dim}"):
        coords, values = np.loadtxt(filename).T
        profile = Profile(values=values, coords=coords)
        profiles[dim].append(profile)


# Fit covariance matrix
# --------------------------------------------------------------------------------------

moments_xx = np.loadtxt(os.path.join(input_dirs["data"], "moments_x.dat"))
moments_yy = np.loadtxt(os.path.join(input_dirs["data"], "moments_y.dat"))
moments_uu = np.loadtxt(os.path.join(input_dirs["data"], "moments_u.dat"))
moments_xy = compute_cov_xy(moments_xx, moments_yy, moments_uu)
moments = np.vstack([moments_xx, moments_yy, moments_xy]).T

cov_matrix, _ = fit_cov(moments, tmats)
norm_matrix = normalization_matrix(cov_matrix, scale=True)
unnorm_matrix = np.linalg.inv(norm_matrix)


# Set up MENT model
# --------------------------------------------------------------------------------------

ndim = 4

transforms = []
for tmat in tmats:
    transform = ment.sim.LinearTransform(tmat)
    transforms.append(transform)

projections = []
for index, transform in enumerate(transforms):
    coords_x = np.copy(profiles["x"][index].coords)
    coords_y = np.copy(profiles["y"][index].coords)
    coords_u = np.copy(profiles["u"][index].coords)

    values_x = np.copy(profiles["x"][index].values)
    values_y = np.copy(profiles["y"][index].values)
    values_u = np.copy(profiles["u"][index].values)

    projection_x = ment.Histogram1D(coords=coords_x, values=values_x, axis=0)
    projection_y = ment.Histogram1D(coords=coords_y, values=values_y, axis=2)

    direction = np.array([+1.0, 0.0, -1.0, 0.0]) / np.sqrt(2.0)
    projection_u = ment.Histogram1D(
        coords=coords_u, values=values_u, direction=direction, axis=None
    )

    projection_x.normalize()
    projection_y.normalize()
    projection_u.normalize()

    projections.append([projection_x, projection_y, projection_u])


prior = ment.prior.GaussianPrior(ndim=ndim, scale=1.0)

sampler = ment.samp.GridSampler(
    grid_limits=(ndim * [(-args.samp_xmax, args.samp_xmax)]),
    grid_shape=tuple(ndim * [args.samp_res]),
    noise=args.samp_noise,
)

model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    unnorm_matrix=unnorm_matrix,
    prior=prior,
    sampler=sampler,
    interpolation_kws=dict(method="linear"),
    mode="sample",
    nsamp=args.samp,
    verbose=True,
)


# Train MENT model
# --------------------------------------------------------------------------------------

trainer = ment.train.Trainer(
    model,
    plot_func=None,
    eval_func=ment.train.Evaluator(args.samp),
    output_dir=output_dir,
)

trainer.train(epochs=args.epochs, learning_rate=args.lr)
