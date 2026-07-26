MENT
==================

MENT is a Python package for reconstructing probability distributions from
projection data using the method of maximum entropy. Its primary application is
phase space tomography in particle accelerators.

A finite set of projections constrains a probability distribution but generally
does not determine it uniquely. MENT finds the compatible distribution that
maximizes relative entropy with respect to a prior distribution.

.. math::

   S[p(x), q(x)] =
   - \int p(x) \log \left( \frac{p(x)}{q(x)} \right) dx

where :math:`p(x)` is the reconstructed distribution, :math:`q(x)` is the prior,
and the integral is over the full space.

Overview
--------

MENT provides both reverse-mode and forward-mode reconstruction workflows.

Reverse mode
   Uses numerical integration and is typically best suited to low-dimensional
   reconstruction problems.

Forward mode
   Uses particle sampling from an unnormalized distribution and is typically
   better suited to high-dimensional problems.

The package also includes covariance fitting utilities, prior distributions,
diagnostics, simulation helpers, and sampling tools.

Installation
------------

Install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/austin-hoover/ment.git
   cd ment
   pip install -e .

To install dependencies for examples and plotting:

.. code-block:: bash

   pip install -e ".[test]"

To install documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Building the Documentation
--------------------------

From the repository root, run:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Or, if using the included ``docs/Makefile``:

.. code-block:: bash

   cd docs
   make html

The generated HTML documentation will be available at:

.. code-block:: text

   docs/_build/html/index.html

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   api

References
----------

1. G. Minerbo, *MENT: A Maximum Entropy Algorithm for Reconstructing a Source
   from Projection Data*, Computer Graphics and Image Processing 10, 48, 1979.

2. G. N. Minerbo, O. R. Sander, and R. A. Jameson,
   *Four-Dimensional Beam Tomography*, IEEE Transactions on Nuclear Science 28,
   2231, 1981.

3. J. C. Wong, A. Shishlo, A. Aleksandrov, Y. Liu, and C. Long,
   *4D Transverse Phase Space Tomography of an Operational Hydrogen Ion Beam via
   Noninvasive 2D Measurements Using Laser Wires*, Physical Review Accelerators
   and Beams 25, 042801, 2022.

4. A. Hoover, *Four-dimensional phase space tomography from one-dimensional
   measurements of a hadron beam*, Physical Review Accelerators and Beams 27,
   122802, 2024.

5. A. Hoover and J. Wong, *High-dimensional maximum-entropy phase space
   tomography using normalizing flows*, Physical Review Research 6, 033163,
   2024.

6. A. Hoover, *N-dimensional maximum-entropy tomography via particle sampling*,
   Physical Review Accelerators and Beams 28, L084601, 2025.
