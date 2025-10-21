# MENT

This repository implements MENT, an algorithm to reconstruct a distribution from its projections using the method of maximum entropy. The primary application of this algorithm is to phase space tomography in particle accelerators.


## Background

A probability distribution is constrained, but not determined, by a finite set of its projections. MENT finds the distribution $p(x)$ that is compatible with the projections and maximizes the relative entropy

```math
S[p(x), q(x)] = - \int p(x) \log \left( \frac{p(x)}{q(x)} \right) dx,
```

where $q(x)$ is considered as a prior over $x$ and the integration is over all space. The constrained maximum-entropy distribution is as simple as possible relative to $q(x)$. This is illustrated in the following figure, which reconstructs a concentric rings distribution from only a few projections. The third row shows another reconstruction that matches the data but is farther from the Gaussian prior.

<img src="docs/images/fig_rings.png" width="400px">


## Implementation

MENT uses the method of Lagrange Multipliers combined with a nonlinear Gauss-Seidel relaxation method to solve the constrained optimization problem. There are two equivalent ways to run the algorithm. The first, called "reverse mode", uses numerical integration; the second, called "forward mode" uses particle sampling. Numerical integration is the best choice in low-dimensional problems, while particle sampling is the better choice in high-dimensional problems.

This repository contains both a forward-mode and reverse-mode implementation of MENT. In forward mode, one must sample particles from an unnormalized distribution function. A grid-based sampler is included for problems of dimension $N <= 4$. Several MCMC algorithms are included for $N >= 4$, including the No-Underrun Sampler (NURS) and Metropolis-Hastings (MH). The samplers are vectorized and can be run with many chains at once.

Each projection is defined as a sum over one or more axes after a transformation of the coordinates. The only requirement on the transformations is that they must be deterministic and one-to-one. The code is set up to take arbitrary transformation functions as inputs. This allows straightforward integration with particle tracking codes.

We also include routines to fit an $N \times N$ covariance matrix to measured projections, which is often a first step before running MENT.


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

Several examples are included in the [examples](https://github.com/austin-hoover/ment/tree/main/examples) folder. These examples demonstrate convergence on a variety of 2D, 4D, and 6D distributions.



## References

[1] G. Minerbo, [MENT: A Maximum Entropy Algorithm for Reconstructing a Source from Projection Data](https://www-sciencedirect-com.ornl.idm.oclc.org/science/article/pii/0146664X79900340), Computer Graphics and Image Processing 10, 48 (1979).

[2] G. N. Minerbo, O. R. Sander, and R. A. Jameson, [Four-Dimensional Beam Tomography](https://ieeexplore.ieee.org/document/4331646), IEEE Transactions on Nuclear Science 28, 2231 (1981).

[3] J. C. Wong, A. Shishlo, A. Aleksandrov, Y. Liu, and C. Long, [4D Transverse Phase Space Tomography of an Operational Hydrogen Ion Beam via Noninvasive 2D Measurements Using Laser Wires](https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.25.042801), Phys. Rev. Accel. Beams 25, 042801 (2022).

[4] A. Hoover, [Four-dimensional phase space tomography from one-dimensional measurements of a hadron beam](https://doi.org/10.1103/PhysRevAccelBeams.27.122802), Physical Review Accelerators and Beams 27, 122802 (2024).

[5] A. Hoover and J. Wong, [High-dimensional maximum-entropy phase space tomography using normalizing flows](https://doi.org/10.1103/PhysRevResearch.6.033163), Physical Review Research 6.3, 033163 (2024).

[6] A. Hoover, [N-dimensional maximum-entropy tomography via particle sampling](https://doi.org/10.1103/zl2h-3v32), Phys. Rev. Accel. Beams 28, L084601 (2025).

[7] A. Hoover, [High-dimensional phase space tomography](https://prebys.physics.ucdavis.edu/NAPAC-25/proceedings/pdf/TUZN01.pdf), Proceedings of NAPAC 2025.
