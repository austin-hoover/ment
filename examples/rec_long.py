"""Reconstruct longitudinal phase space distribution from turn-by-turn projections.

This script uses a PyORBIT [https://github.com/PyORBIT-Collaboration/PyORBIT3] lattice 
model consisting of a harmonic RF cavity surrounded by two drifts. Things are a bit slow
because we have to repeatedly convert between NumPy arrays and Bunch objects, but it works.

Note that one MENT iteration requires simulating all projectionos. If projectiono k
is measured after k turns, then we must first track the bunch 1 turn, then resample
and track 2 turns, then resample and track 3 turns, etc. In total, we must track
n * (n + 1) / 2 turns. For a significant number of turns, ART may be the better 
option.
"""
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import ment

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.rf_cavities import Harmonic_RFNode
from orbit.teapot import DriftTEAPOT
from orbit.teapot import TEAPOT_Ring


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


# Forward model
# --------------------------------------------------------------------------------------

def get_part_coords(bunch: Bunch, index: int) -> list[float]:
    x = bunch.x(index)
    y = bunch.y(index)
    z = bunch.z(index)
    xp = bunch.xp(index)
    yp = bunch.yp(index)
    de = bunch.dE(index)
    return [x, xp, y, yp, z, de]


def set_part_coords(bunch: Bunch, index: int, coords: list[float]) -> Bunch:
    (x, xp, y, yp, z, de) = coords
    bunch.x(index, x)
    bunch.y(index, y)
    bunch.z(index, z)
    bunch.xp(index, xp)
    bunch.yp(index, yp)
    bunch.dE(index, de)
    return bunch


def get_bunch_coords(bunch: Bunch, axis: tuple[int, ...] = None) -> np.ndarray:
    x = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        x[i, :] = get_part_coords(bunch, i)
    if axis is not None:
        x = x[:, axis]
    return x


def set_bunch_coords(bunch: Bunch, x: np.ndarray, axis: tuple[int, ...] = None) -> Bunch:
    if axis is None:
        axis = tuple(range(6))

   # Resize
    size = x.shape[0]
    size_error = size - bunch.getSize()
    if size_error > 0:
        for _ in range(size_error):
            bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        for i in range(size, bunch.getSize()):
            bunch.deleteParticleFast(i)
        bunch.compress()

    for i in range(bunch.getSize()):
        coords = get_part_coords(bunch, i)
        for j in range(len(axis)):
            coords[axis[j]] = x[i, j]
        bunch = set_part_coords(bunch, i, coords)
    return bunch


class ORBITTransform:
    def __init__(
        self,
        lattice: AccLattice,
        bunch: Bunch,
        axis: tuple[int, ...],
        nturns: int = 1,
    ) -> None:
        self.lattice = lattice
        self.bunch = bunch
        self.axis = axis
        self.nturns = nturns

    def track_bunch(self) -> Bunch:
        bunch = Bunch()
        self.bunch.copyBunchTo(bunch)
        for _ in range(self.nturns):
            self.lattice.trackBunch(bunch)
        return bunch

    def __call__(self, x: np.ndarray) -> np.ndarray:
        set_bunch_coords(self.bunch, x, axis=self.axis)
        bunch = self.track_bunch()
        x_out = get_bunch_coords(bunch, axis=self.axis)
        return x_out
    

# Create accelerator lattice (drift, rf, drift)
drift_node_1 = DriftTEAPOT()
drift_node_2 = DriftTEAPOT()
drift_node_1.setLength(124.0)
drift_node_2.setLength(124.0)


z_to_phi = 2.0 * np.pi / 248.0
rf_hnum = 1.0
rf_length = 0.0
rf_synchronous_de = 0.0
rf_voltage = 300.0e-06
rf_phase = 0.0
rf_node = Harmonic_RFNode(z_to_phi, rf_synchronous_de, rf_hnum, rf_voltage, rf_phase, rf_length)

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


# Create transform functions
turn_min = 0
turn_max = 500
turn_step = int((turn_max - turn_min) / nmeas)
turns = list(range(turn_min, turn_max + turn_step, turn_step))

transforms = []
for nturns in turns:
    transform = ORBITTransform(lattice, bunch, nturns=nturns, axis=(4, 5))
    transforms.append(transform)

limits = [
    (-0.5 * lattice.getLength(), +0.5 * lattice.getLength()), 
    (-0.030, 0.030)
]
    
# Create a list of histogram diagnostics for each transform.
bin_edges = np.linspace(limits[0][0], limits[0][1], 100)
diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])


# Training data
# --------------------------------------------------------------------------------------

# Here we simulate the projections; in real life the projections would be measured.
projections = ment.sim.simulate(x_true, transforms, diagnostics)


# Reconstruction model
# --------------------------------------------------------------------------------------

# Define prior distribution for relative entropy calculation
prior = ment.prior.GaussianPrior(ndim=2, scale=[200.0, 0.020])

# Define particle sampler (if mode="sample")
sampler = ment.samp.GridSampler(
    grid_limits=limits,
    grid_shape=(128, 128),
)

# Set up MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    mode="sample",
    verbose=2,
)


# Training
# --------------------------------------------------------------------------------------


def plot_model(model):
    # Sample particles
    x_pred = model.sample(size)

    # Plot sim vs. measured profiles
    projections_true = ment.sim.copy_histograms(model.projections)
    projections_true = ment.utils.unravel(projections_true)

    projections_pred = ment.sim.copy_histograms(model.diagnostics)
    projections_pred = ment.sim.simulate(x_pred, transforms, projections_pred)
    projections_pred = ment.utils.unravel(projections_pred)

    fig, axs = plt.subplots(
        ncols=nmeas, figsize=(11.0, 1.0), sharey=True, sharex=True, constrained_layout=True
    )
    for i, ax in enumerate(axs):
        values_pred = projections_pred[i].values
        values_true = projections_true[i].values
        ax.plot(values_pred / values_true.max(), color="lightgray")
        ax.plot(values_true / values_true.max(), color="black", lw=0.0, marker=".", ms=2.0)
    return fig


for epoch in range(4):
    print("epoch =", epoch)
    
    if epoch > 0:
        model.gauss_seidel_step(learning_rate=0.90)

    fig = plot_model(model)
    fig.savefig(os.path.join(output_dir, f"fig_proj_{epoch:02.0f}.png"))
    plt.close()

# Plot final distribution
x_pred = model.sample(x_true.shape[0])

fig, axs = plt.subplots(ncols=2, constrained_layout=True)
for ax, x in zip(axs, [x_pred, x_true]):
    ax.hist2d(x[:, 0], x[:, 1], bins=100, range=limits)
fig.savefig(os.path.join(output_dir, f"fig_dist_{epoch:02.0f}.png"))
plt.close()

