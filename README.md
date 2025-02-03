# MENT

This repository implements MENT, an algorithm to reconstruct a distribution from its projections using the method of maximum entropy. The primary application of this algorithm is to phase space tomography in particle accelerators.


## Background

A probability distribution is constrained, but not determined, by a finite set of its projections. Given a set of projections, MENT finds the unique distribution $\rho(x)$ that is compatible with the projections while maximizing the relative entropy
$$
S[\rho(x), \rho_*(x)] = - \int \rho(x) \log \left( \frac{\rho(x)}{\rho_*(x)} \right) dx,
$$
where $\rho_*(x)$ is considered as a prior over $x$ and the integration is over all space. When the constraints provided by the measurements are not tight, the additional prior information pulls the reconstruction toward the prior. This is illustrated in the following figure, which reconstructs a concentric rings distribution from only a few projections wit a Gaussian prior. The third row shows another reconstruction that matches the data but is farther from the prior.

<img src="docs/images/fig_rings.png" width="400px">


## Implementation

MENT uses the method of Lagrange Multipliers combined with a nonlinear Gauss-Seidel relaxation method to solve the constrained optimization problem. There are two equivalent ways to run the algorithm. The first, called "reverse mode", uses numerical integration; the second, called "forward mode" uses particle sampling, i.e., MCMC. Numerical integration is the best choice in low-dimensional problems, while particle sampling is the better choice in high-dimensional problems.

This repository contains both a forward-mode and reverse-mode implementation of MENT. In forward mode, one must sample particles from an unnormalized distribution function. An accurate grid-based sampler is included for problems of dimension $N <= 4$, and an MCMC Metropolis Hastings sampler is included for problems of dimension $N > 4$.

Each projection is defined as a sum over one or more axes *after* a transformation of the coordinates. The only requirement on the transformations is that they must be deterministic and one-to-one. The code is set up to take arbitrary transformation functions as inputs; these functions map NumPy arrays to NumPy arrays. This allows straightforward integration with, i.e., beam physics simulation codes.


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


## Examples

Several examples are included in the [examples](https://github.com/austin-hoover/ment/tree/main/examples) folder. See `examples/simple.py` for the basic setup.
Documentation in progress.

```python
import numpy as np
import matplotlib.pyplot as plt

import ment


# Settings
ndim = 2
nmeas = 7
seed = 0


# Ground truth distribution
# --------------------------------------------------------------------------------------

rng = np.random.default_rng(seed)
x_true = rng.normal(size=(100_000, ndim))
x_true = x_true / np.linalg.norm(x_true, axis=1)[:, None]
x_true = x_true * 1.5
x_true = x_true + rng.normal(size=x_true.shape, scale=0.25)


# Forward model
# --------------------------------------------------------------------------------------

# Create a list of transforms. Each transform is a function with the call signature
# `transform(x: np.ndarray) -> np.ndarray`.


def rotation_matrix(angle: float) -> np.ndarray:
    M = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    M = np.array(M)
    return M


transforms = []
for angle in np.linspace(0.0, np.pi, nmeas, endpoint=False):
    M = rotation_matrix(angle)
    transform = ment.sim.LinearTransform(M)
    transforms.append(transform)

# Create a list of histogram diagnostics for each transform.
bin_edges = np.linspace(-4.0, 4.0, 55)
diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.Histogram1D(axis=0, edges=bin_edges)
    diagnostics.append([diagnostic])


# Training data
# --------------------------------------------------------------------------------------

# Here we simulate the projections; in real life the projections would be
# measured.
projections = ment.sim.copy_histograms(diagnostics)
projections = ment.sim.simulate(x_true, transforms, diagnostics)


# Reconstruction model
# --------------------------------------------------------------------------------------

# Define prior distribution for relative entropy calculation
prior = ment.prior.GaussianPrior(ndim=2, scale=[1.0, 1.0])

# Define particle sampler (if mode="sample")
sampler = ment.samp.GridSampler(
    grid_limits=(2 * [(-4.0, 4.0)]),
    grid_shape=(128, 128),
)

# Define integration grid (if mode="integrate"). You need separate integration
# limits for each measurement.
integration_limits = [
    (-4.0, 4.0),
]
integration_limits = [[integration_limits for _ in diagnostics] for _ in transforms]
integration_size = 100

# Set up MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    integration_limits=integration_limits,
    integration_size=integration_size,
    mode="integrate",
)


# Training
# --------------------------------------------------------------------------------------


def plot_model(model):
    # Sample particles
    x_pred = model.sample(100_000)

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
    plt.show()


for epoch in range(4):
    if epoch > 0:
        model.gauss_seidel_step(learning_rate=0.90)

    plot_model(model)


# Plot final distribution
x_pred = model.sample(1_000_000)

fig, axs = plt.subplots(ncols=2, constrained_layout=True)
for ax, X in zip(axs, [x_pred, x_true]):
    ax.hist2d(X[:, 0], X[:, 1], bins=55, range=[(-4.0, 4.0), (-4.0, 4.0)])
    ax.set_aspect(1.0)
plt.show()
```


## References

[1] G. Minerbo, [MENT: A Maximum Entropy Algorithm for Reconstructing a Source from Projection Data](https://www-sciencedirect-com.ornl.idm.oclc.org/science/article/pii/0146664X79900340), Computer Graphics and Image Processing 10, 48 (1979).

[2] G. N. Minerbo, O. R. Sander, and R. A. Jameson, [Four-Dimensional Beam Tomography](https://ieeexplore.ieee.org/document/4331646), IEEE Transactions on Nuclear Science 28, 2231 (1981).

[3] J. C. Wong, A. Shishlo, A. Aleksandrov, Y. Liu, and C. Long, [4D Transverse Phase Space Tomography of an Operational Hydrogen Ion Beam via Noninvasive 2D Measurements Using Laser Wires](https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.25.042801), Phys. Rev. Accel. Beams 25, 042801 (2022).

[4] A. Hoover, [Four-dimensional phase space tomography from one-dimensional measurements of a hadron beam](https://doi.org/10.1103/PhysRevAccelBeams.27.122802), Physical Review Accelerators and Beams 27, 122802 (2024).

[5] A. Hoover and J. Wong, [High-dimensional maximum-entropy phase space tomography using normalizing flows](https://doi.org/10.1103/PhysRevResearch.6.033163), Physical Review Research 6.3, 033163 (2024).

[6] A. Hoover, [N-dimensional maximum-entropy tomography via particle sampling](https://arxiv.org/abs/2409.17915), arXiv preprint arXiv:2409.17915 (2024).
