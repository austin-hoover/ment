import os
import time
import typing
from pprint import pprint
from typing import Any
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import psdist as ps
import psdist.plot as psv

from ..core import MENT
from ..sim import forward
from ..sim import forward_with_diag_update
from ..utils import unravel


class Plotter:
    """Plots predicted distribution and projections."""

    def __init__(
        self,
        n_samples: int,
        plot_proj: list[Callable],
        plot_dist: list[Callable],
    ) -> None:
        """Constructor.

        Parameters
        ----------
        n_samples : int
            Number of samples to plot.
        plot_dist: list[callable]
            Plots samples from true and predicted distributions.
            Signature: `plot_dist(X)`.
        plot_proj: list[callable]
            Plots simulated vs. actual measurements.
            Signature: `plot_proj(values_meas, values_pred, coords)`.
        """
        self.n_samples = n_samples
        self.plot_proj = plot_proj
        self.plot_dist = plot_dist

        if self.plot_proj is not None:
            if type(self.plot_proj) not in [list, tuple]:
                self.plot_proj = [self.plot_proj]
        if self.plot_dist is not None:
            if type(self.plot_dist) not in [list, tuple]:
                self.plot_dist = [self.plot_dist]

    def __call__(self, model: MENT) -> list:
        # Generate particles.
        x_pred = model.unnormalize(model.sample(self.n_samples))

        # Simulate measurements.
        projections_true = model.projections
        projections_true = unravel(projections_true)
        projections_pred = forward_with_diag_update(
            x_pred,
            model.transforms,
            model.diagnostics,
            kde=False,
            blur=False,
        )
        projections_pred = unravel(projections_pred)
        diagnostics = unravel(model.diagnostics)

        figs = []

        # Plot samples.
        if self.plot_dist is not None:
            for function in self.plot_dist:
                fig, axs = function(x_pred)
                figs.append(fig)

        # Plot measured vs. simulated projections.
        if self.plot_proj is not None:
            coords = [diagnostic.coords for diagnostic in diagnostics]
            for function in self.plot_proj:
                fig, axs = function(projections_pred, projections_true, coords)
                figs.append(fig)
        return figs


class PlotProj1D:
    def __init__(
        self,
        ncols: int = 7,
        ymin: float = None,
        ymax: float = None,
        xmin: float = None,
        xmax: float = None,
        xlim_scale: float = None,
        log: bool = False,
        fig_kws: dict = None,
        plot_kws_pred: dict = None,
        plot_kws_true: dict = None,
        **plot_kws,
    ) -> None:
        self.ncols = ncols
        self.log = log
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xlim_scale = xlim_scale

        if self.log:
            if self.ymax is None:
                self.ymax = 5.0
            if self.ymin is None:
                self.ymin = 1.00e-05
        else:
            if self.ymax is None:
                self.ymax = 1.25

        self.plot_kws = plot_kws

        self.plot_kws_pred = plot_kws_pred
        if self.plot_kws_pred is None:
            self.plot_kws_pred = {}
        self.plot_kws_pred.setdefault("color", "lightgray")
        self.plot_kws_pred.setdefault("lw", None)

        self.plot_kws_true = plot_kws_true
        if self.plot_kws_true is None:
            self.plot_kws_true = {}
        self.plot_kws_true.setdefault("color", "black")
        self.plot_kws_true.setdefault("lw", 0.0)
        self.plot_kws_true.setdefault("marker", ".")
        self.plot_kws_true.setdefault("ms", 1.0)

        for key, val in self.plot_kws.items():
            self.plot_kws_pred[key] = val
            self.plot_kws_true[key] = val

        self.fig_kws = fig_kws
        if self.fig_kws is None:
            self.fig_kws = {}
        self.fig_kws.setdefault("sharex", False)
        self.fig_kws.setdefault("sharey", True)

    def __call__(
        self,
        projections_pred: list[np.ndarray],
        projections_true: list[np.ndarray],
        coords: list[np.ndarray],
    ) -> tuple:

        nmeas = len(projections_pred)
        ncols = min(self.ncols, nmeas)
        nrows = int(np.ceil(nmeas / ncols))

        self.fig_kws.setdefault("ncols", ncols)
        self.fig_kws.setdefault("nrows", nrows)
        self.fig_kws.setdefault("figsize", (1.50 * ncols, 1.25 * nrows))

        fig, axs = pplt.subplots(**self.fig_kws)

        for i, (y_pred, y_true) in enumerate(zip(projections_pred, projections_true)):
            ax = axs[i]
            psv.plot_profile(y_pred / np.max(y_true), coords=coords[i], ax=ax, **self.plot_kws_pred)
            psv.plot_profile(y_true / np.max(y_true), coords=coords[i], ax=ax, **self.plot_kws_true)
            ax.format(ymin=self.ymin, ymax=self.ymax, xmin=self.xmin, xmax=self.xmax)
            if self.log:
                ax.format(yscale="log", yformatter="log")

        if self.xlim_scale is not None:
            for ax in axs:
                xlim = np.array(ax.get_xlim())
                xlim = xlim * self.xlim_scale
                ax.format(xlim=xlim)
        return (fig, axs)


class PlotProj2D_Contour:
    def __init__(
        self, 
        ncols_max: int = 7,
        lim_share: bool = False,
        lim_scale: float = 1.0,
        plot_kws_true: dict = None,
        plot_kws_pred: dict = None,
        **plot_kws
    ) -> None:
        self.ncols_max = ncols_max
        self.lim_share = lim_share
        self.lim_scale = lim_scale

        self.plot_kws = plot_kws

        self.plot_kws_true = plot_kws_true
        if self.plot_kws_true is None:
            self.plot_kws_true = {}

        self.plot_kws_pred = plot_kws_true
        if self.plot_kws_pred is None:
            self.plot_kws_pred = {}

        self.plot_kws.setdefault("kind", "contour")
        self.plot_kws.setdefault("levels", np.linspace(0.01, 1.0, 7))
        self.plot_kws.setdefault("lw", 0.75)
        self.plot_kws.setdefault("process_kws", {})
        self.plot_kws["process_kws"].setdefault("norm", "max")
        self.plot_kws["process_kws"].setdefault("blur", 1.0)
        
        self.plot_kws_true.setdefault("color", "black")
        self.plot_kws_pred.setdefault("color", "red")

        for key, val in self.plot_kws.items():
            self.plot_kws_true[key] = val
            self.plot_kws_pred[key] = val
    
    def __call__(
        self,
        projections_pred: list[np.ndarray],
        projections_true: list[np.ndarray],
        coords_list: list[np.ndarray],
    ) -> tuple:
        nmeas = len(projections_true)
        ncols = min(nmeas, self.ncols_max)
        nrows = int(np.ceil(nmeas / ncols))
        
        fig, axs = pplt.subplots(ncols=ncols, nrows=nrows, figwidth=(1.1 * ncols), share=self.lim_share)
        for index in range(nmeas):
            values_true = projections_true[index]
            values_pred = projections_pred[index]
            coords = coords_list[index]
            ax = axs[index]
            psv.plot_image(values_true, coords=coords, ax=ax, **self.plot_kws_true)
            psv.plot_image(values_pred, coords=coords, ax=ax, **self.plot_kws_pred)
            
        if self.lim_share:
            ax = axs[0]
            axs.format(xlim=np.multiply(ax.get_xlim(), self.lim_scale))
            axs.format(ylim=np.multiply(ax.get_ylim(), self.lim_scale))
        else:
            for ax in axs:
                ax.format(xlim=np.multiply(ax.get_xlim(), self.lim_scale))
                ax.format(ylim=np.multiply(ax.get_ylim(), self.lim_scale))
            
        return fig, axs


class PlotDistCorner:
    def __init__(
        self,
        log: bool = False,
        fig_kws: dict = None,
        diag_ymin: float = None,
        diag_ymax: float = None,
        **plot_kws,
    ) -> None:
        self.log = log

        self.fig_kws = fig_kws
        if self.fig_kws is None:
            self.fig_kws = {}

        self.plot_kws = plot_kws
        self.plot_kws.setdefault("bins", 64)
        self.plot_kws.setdefault("mask", False)
        self.plot_kws.setdefault("cmap", "viridis")

        self.diag_ymin = diag_ymin
        self.diag_ymax = diag_ymax

        if self.log:
            if self.diag_ymin is None:
                self.diag_ymin = 1.00e-05
            if self.diag_ymax is None:
                self.diag_ymax = 5.0

        if self.log:
            self.plot_kws["norm"] = "log"

    def __call__(self, X: np.ndarray) -> tuple:
        grid = psv.CornerGrid(X.shape[1], **self.fig_kws)
        grid.plot_points(X, **self.plot_kws)

        grid.format_diag(ymax=self.diag_ymax, ymin=self.diag_ymin)
        if self.log:
            grid.format_diag(yscale="log", yformatter="log")

        return (grid.fig, grid.axs)
