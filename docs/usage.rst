Usage Guide
===========

Workflow
--------

A typical MENT workflow consists of the following steps:

1. Define or load measured projection data.
2. Choose a prior distribution.
3. Define transformations associated with each projection.
4. Optionally fit a covariance matrix from measured projections.
5. Run MENT in reverse mode or forward mode.
6. Compare reconstructed projections against the measured data.

Reconstruction Modes
--------------------

Reverse Mode
~~~~~~~~~~~~

Reverse mode uses numerical integration. It is usually the most practical option
for low-dimensional problems where integration over the reconstruction grid is
tractable.

Forward Mode
~~~~~~~~~~~~

Forward mode uses particle sampling. It is usually preferred for
higher-dimensional problems where direct numerical integration becomes
expensive.

Projection Model
----------------

Each projection is defined after applying a coordinate transformation. The
transformations should be deterministic and one-to-one. This makes it possible
to use MENT with externally supplied transformation functions, including
transformations from accelerator tracking workflows.

Covariance Fitting
------------------

MENT includes utilities for fitting covariance matrices to measured projection
data. This is often useful as an initialization step before full maximum-entropy
reconstruction.

Examples
--------

The repository includes examples for several reconstruction scenarios, including
low-dimensional reconstructions, higher-dimensional reconstructions, covariance
fitting, and sampling-based workflows.

After installing the optional example dependencies, example scripts can be run
from the repository root:

.. code-block:: bash

   pip install -e ".[test]"
   python examples/rec_2d.py
