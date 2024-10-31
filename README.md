# MENT

This repository implements MENT, an algorithm to reconstruct a distribution from its projections using the method of maximum entropy.


## Installation

```
git clone https://github.com/austin-hoover/ment.git
cd ment
pip install -e .
```

To run examples using built-in plotting functions:
```
pip install -e '.[test]'
```


## References

1. G. Minerbo, [MENT: A Maximum Entropy Algorithm for Reconstructing a Source from Projection Data](https://www-sciencedirect-com.ornl.idm.oclc.org/science/article/pii/0146664X79900340), Computer Graphics and Image Processing 10, 48 (1979).
2. G. N. Minerbo, O. R. Sander, and R. A. Jameson, [Four-Dimensional Beam Tomography](https://ieeexplore.ieee.org/document/4331646), IEEE Transactions on Nuclear Science 28, 2231 (1981).
3. J. C. Wong, A. Shishlo, A. Aleksandrov, Y. Liu, and C. Long, [4D Transverse Phase Space Tomography of an Operational Hydrogen Ion Beam via Noninvasive 2D Measurements Using Laser Wires](https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.25.042801), Phys. Rev. Accel. Beams 25, 042801 (2022).
4. A. Hoover, [Four-dimensional phase space tomography from one-dimensional measurements of a hadron beam](), arXiv preprint arXiv:2409.02862 (2024).
5. A. Hoover and J. Wong, [High-dimensional maximum-entropy phase space tomography using normalizing flows](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033163), Physical Review Research 6.3 (2024): 033163.
6. A. Hoover, [N-dimensional maximum-entropy tomography via particle sampling](https://arxiv.org/abs/2409.17915), arXiv preprint arXiv:2409.17915 (2024).


## Examples

Several examples are included in the `examples` folder. See `examples/simple.py` for the basic setup.
Documentation is still in progress.

```
import numpy as np
import matplotlib.pyplot as plt

import ment


# Settings
ndim = 2
nmeas = 7
seed = 0


# Define forward model
# --------------------------------------------------------------------------------------

# Create a list of transforms. Each transform is a function with the call signature 
# `transform(X)`, where X is a numpy array of particle coordinates (shape (n, d)).
def rotation_matrix(angle: float) -> np.ndarray:
    M = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    M = np.array(M)
    return M

transforms = []
for angle in np.linspace(0.0, np.pi, nmeas, endpoint=False):
    M = rotation_matrix(angle)
    transform = ment.sim.LinearTransform(M)
    transforms.append(transform)

# Create a list of diagnostics to apply after each transform. The call signature of
# each diagnostic is `diagnostic(X)`, which generates a histogram.
bin_edges = np.linspace(-4.0, 4.0, 55)
diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])


# Generate training data
# --------------------------------------------------------------------------------------

# Create initial distribution
rng = np.random.default_rng(seed)
X_true = rng.normal(size=(100_000, ndim))
X_true = X_true / np.linalg.norm(X_true, axis=1)[:, None]
X_true = X_true * 1.5
X_true = X_true + rng.normal(size=X_true.shape, scale=0.25)

# Assign measured profiles to each diagnostic. Here we simulate the measurements 
# using the true distribution. You can call `ment.sim.forward(X, transforms, diagnostics)`
# instead of the following loop:
projections = []
for index, transform in enumerate(transforms):
    U_true = transform(X_true)
    projections.append([diagnostic(U_true) for diagnostic in diagnostics[index]])


# Create reconstruction model
# --------------------------------------------------------------------------------------

# Define prior distribution for relative entropy
prior = ment.prior.GaussianPrior(ndim=2, scale=[1.0, 1.0])

# Define particle sampler (if mode="sample")
sampler = ment.samp.GridSampler(
    grid_limits=(2 * [(-4.0, 4.0)]),
    grid_shape=(128, 128),
)

# Define integration grid (if mode="integrate"). You need separate integration
# limits for each measurement.
integration_limits = [(-4.0, 4.0),]
integration_limits = [[integration_limits for _ in diagnostics] for _ in transforms]
integration_size = 100

# Set up MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    diagnostics=diagnostics,
    projections=projections,
    prior=prior,
    sampler=sampler,
    integration_limits=integration_limits,
    integration_size=integration_size,
    mode="integrate",
)


# Train the model
# --------------------------------------------------------------------------------------

def plot_model(model):
    # Sample particles
    X_pred = model.sample(100_000)

    # Plot distribution
    fig, axs = plt.subplots(ncols=2, constrained_layout=True)
    for ax, X in zip(axs, [X_pred, X_true]):
        ax.hist2d(X[:, 0], X[:, 1], bins=55, range=[(-4.0, 4.0), (-4.0, 4.0)])
        ax.set_aspect(1.0)
    plt.show()

    # Plot sim vs. measured profiles
    projections_true = model.projections
    projections_pred = ment.sim.forward(X_pred, model.transforms, model.diagnostics)
    projections_true = ment.utils.unravel(projections_true)
    projections_pred = ment.utils.unravel(projections_pred)

    fig, axs = plt.subplots(ncols=nmeas, figsize=(11.0, 1.0), sharey=True, sharex=True)
    for i, ax in enumerate(axs):
        values_pred = projections_pred[i]
        values_true = projections_true[i]
        ax.plot(values_pred / values_true.max(), color="gray")
        ax.plot(values_true / values_true.max(), color="black", lw=0.0, marker=".", ms=2.0)
    plt.show()


for epoch in range(4):
    if epoch > 0:
        model.gauss_seidel_step(learning_rate=0.90)
        
    plot_model(model)
```
