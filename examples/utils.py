import os
import time

import numpy as np
import psdist as ps
import psdist.plot as psv
import ultraplot as plt

import ment


def plot_corner_upper_lower(
    x_pred: np.ndarray, x_true: np.ndarray, n_bins: int, limits: list[tuple[float, float]]
):
    nsamp, ndim = x_pred.shape

    grid = psv.CornerGrid(ndim=ndim, figwidth=(ndim * 1.25), corner=False)
    kws = dict(limits=limits, bins=n_bins, mask=True)
    grid.plot_points(
        x_true[:nsamp, :],
        lower=False,
        diag_kws=dict(kind="step", color="red8", lw=1.25),
        cmap=psv.cubehelix_cmap(color="red"),
        **kws,
    )
    grid.plot_points(
        x_pred[:nsamp, :],
        upper=False,
        diag_kws=dict(kind="step", color="blue8", lw=1.25),
        cmap=psv.cubehelix_cmap(color="blue"),
        **kws,
    )
    return grid.axs
