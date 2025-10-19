"""Reconstruct longitudinal phase space distribution from turn-by-turn projections.

This script uses a PyORBIT [https://github.com/PyORBIT-Collaboration/PyORBIT3] lattice
model consisting of a harmonic RF cavity surrounded by two drifts. Things are a bit slow
because we have to repeatedly convert between NumPy arrays and Bunch objects, but it works.

Note that one MENT iteration requires simulating all projections. If projections are
measured after k turns, then we must first track the bunch 1 turn, then resample
and track 2 turns, then resample and track 3 turns, etc. In total, we must track
n * (n + 1) / 2 turns. For a significant number of turns, ART may be the better
option.

This example also spends considerable time converting between NumPy arrays and
PyORBIT Bunch objects.
"""
import math
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import torch

import ment
from ment.sim.orbit import ORBITTransform
from ment.sim.orbit import get_bunch_coords
from ment.sim.orbit import set_bunch_coords

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.rf_cavities import Harmonic_RFNode
from orbit.teapot import DriftTEAPOT
from orbit.teapot import TEAPOT_Ring

plt.style.use("../style.mplstyle")


# Setup
# --------------------------------------------------------------------------------------

ndim = 2
nmeas = 7
seed = 0
size = 100_000

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Distribution
# --------------------------------------------------------------------------------------

rng = np.random.default_rng(seed)

x_true = np.zeros((size, 2))
x_true[:, 0] = 0.60 * rng.uniform(-124.0, 124.0, x_true.shape[0])
x_true[:, 1] = rng.normal(scale=0.0025, size=x_true.shape[0])
x_true = torch.from_numpy(x_true)


# Forward model
# --------------------------------------------------------------------------------------

# Create accelerator lattice (drift, rf, drift)
drift_node_1 = DriftTEAPOT()
drift_node_2 = DriftTEAPOT()
drift_node_1.setLength(124.0)
drift_node_2.setLength(124.0)

z_to_phi = 2.0 * math.pi / 248.0
rf_hnum = 1.0
rf_length = 0.0
rf_synchronous_de = 0.0
rf_voltage = 300.0e-06
rf_phase = 0.0
rf_node = Harmonic_RFNode(
    z_to_phi, rf_synchronous_de, rf_hnum, rf_voltage, rf_phase, rf_length
)

lattice = TEAPOT_Ring()
lattice.addNode(drift_node_1)
lattice.addNode(rf_node)
lattice.addNode(drift_node_2)
lattice.initialize()


# Create bunch
bunch = Bunch()
bunch.mass(0.938)
bunch.getSyncParticle().kinEnergy(1.000)

for i in range(x_true.shape[0]):
    bunch.addParticle(0.0, 0.0, 0.0, 0.0, x_true[i, 0], x_true[i, 1])


# Evolve forward a few turns; this will be our ground-truth distribution.
for _ in range(250):
    lattice.trackBunch(bunch)

x_true = get_bunch_coords(bunch, axis=(4, 5))


# Create transform functions
turn_min = 0
turn_max = 500
turn_step = int((turn_max - turn_min) / nmeas)
turns = list(range(turn_min, turn_max + turn_step, turn_step))

transforms = []
for nturns in turns:
    transform = ORBITTransform(lattice, bunch, turns=nturns, axis=(4, 5))
    transforms.append(transform)

limits = [(-0.5 * lattice.getLength(), +0.5 * lattice.getLength()), (-0.030, 0.030)]

# Create a list of histogram diagnostics for each transform.
bin_edges = torch.linspace(limits[0][0], limits[0][1], 100)
diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])


# Training data
# --------------------------------------------------------------------------------------

projections = ment.simulate(x_true, transforms, diagnostics)


# Reconstruction model
# --------------------------------------------------------------------------------------

prior = ment.prior.GaussianPrior(ndim=2, scale=[200.0, 0.020])

sampler = ment.samp.GridSampler(
    limits=limits,
    shape=(128, 128),
)

model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    mode="forward",
    verbose=2,
    diag_kws=dict(blur=1.0, thresh=0.001, thresh_type="frac"),
)


# Training
# --------------------------------------------------------------------------------------


def plot_model(model):
    x_pred = model.sample(size)

    projections_true = ment.sim.copy_histograms(model.projections)
    projections_true = ment.utils.unravel(projections_true)

    projections_pred = ment.sim.copy_histograms(model.diagnostics)
    projections_pred = ment.sim.simulate(x_pred, transforms, projections_pred)
    projections_pred = ment.unravel(projections_pred)

    fig, axs = plt.subplots(
        ncols=nmeas,
        figsize=(12.0, 1.25),
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


for iteration in range(4):
    print("iteration =", iteration)

    if iteration > 0:
        model.gauss_seidel_step(learning_rate=0.90)

    fig = plot_model(model)
    fig.savefig(os.path.join(output_dir, f"fig_proj_{iteration:02.0f}.png"))
    plt.close()


# Plot final distribution
x_pred = model.sample(x_true.shape[0])

fig, axs = plt.subplots(ncols=2, figsize=(6, 3), constrained_layout=True)
for ax, x in zip(axs, [x_pred, x_true]):
    ax.hist2d(x[:, 0], x[:, 1], bins=100, range=limits)
fig.savefig(os.path.join(output_dir, f"fig_dist_{iteration:02.0f}.png"))
plt.close()
