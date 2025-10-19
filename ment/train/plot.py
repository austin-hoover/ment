from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from ..core import MENT
from ..diag import Histogram
from ..diag import Histogram1D
from ..diag import HistogramND
from ..sim import simulate
from ..sim import simulate_with_diag_update
from ..sim import copy_histograms
from ..utils import unravel


def process_image(
    values: np.ndarray,
    blur: float = 0.0,
    thresh: float = 0.0,
    thresh_type: str = "abs",
    scale: str = None,
) -> np.ndarray:
    """Process image values (thresh, blur, scale)."""
    values = np.copy(np.array(values))

    if blur:
        values = scipy.ndimage.gaussian_filter(values, blur)

    if thresh:
        min_value = thresh
        if thresh_type == "frac":
            min_value = thresh * np.max(values)
        values[values > min_value] = 0.0

    if scale:
        if scale == "max":
            values = values / np.max(values)
        else:
            raise ValueError(f"Invalid scale {scale}")

    return values


def plot_image(
    values: np.ndarray,
    edges: list[np.ndarray],
    ax=None,
    kind: str = "pcolor",
    proc_kws: dict = None,
    **plot_kws,
) -> None:
    """Plot two-dimensional image."""
    if proc_kws is None:
        proc_kws = {}

    values = process_image(values, **proc_kws)
    coords = [0.5 * (e[:-1] + e[1:]) for e in edges]

    if kind == "pcolor":
        ax.pcolormesh(coords[0], coords[1], values.T, **plot_kws)
    elif kind == "contour":
        ax.contour(coords[0], coords[1], values.T, **plot_kws)
    elif kind == "contourf":
        ax.contourf(coords[0], coords[1], values.T, **plot_kws)
    else:
        raise ValueError(f"Invalid kind {kind}")


def plot_image_1d(
    values: np.ndarray,
    edges: list[np.ndarray],
    ax=None,
    kind: str = "line",
    proc_kws: dict = None,
    **plot_kws,
) -> None:
    """Plot one-dimensional image."""
    if proc_kws is None:
        proc_kws = {}

    values = process_image(values, **proc_kws)
    coords = 0.5 * (edges[:-1] + edges[1:])

    if kind == "stairs":
        plot_kws.pop("marker", None)
        plot_kws.pop("ms", None)
        ax.stairs(values, edges, **plot_kws)
    elif kind == "line":
        ax.plot(coords, values, **plot_kws)
    else:
        raise ValueError(f"Invalid kind {kind}")


def plot_points(
    x: np.ndarray, bins: int, limits: list[tuple[float, float]], ax=None, **plot_kws
) -> None:
    values, edges = np.histogramdd(x, bins=bins, range=limits)
    return plot_image(values=values, edges=edges, ax=ax, **plot_kws)


def plot_points_1d(
    x: np.ndarray, bins: int, limits: tuple[float, float], ax=None, **plot_kws
) -> None:
    values, edges = np.histogram(x, bins=bins, range=limits)
    return plot_image_1d(values=values, edges=edges, ax=ax, **plot_kws)


class CornerGrid:
    def __init__(self, ndim: int, **fig_kws) -> None:
        self.ndim = ndim

        self.fig_kws = fig_kws
        self.fig_kws["ncols"] = ndim
        self.fig_kws["nrows"] = ndim
        self.fig_kws.setdefault("constrained_layout", True)

        self.fig, self.axs = plt.subplots(**self.fig_kws)

        for i in range(self.ndim):
            for j in range(self.ndim):
                if j > i:
                    self.axs[i, j].axis("off")

        for i in range(self.ndim):
            for j in range(self.ndim):
                if j > 0:
                    self.axs[i, j].set_yticks([])
                if i < self.ndim - 1:
                    self.axs[i, j].set_xticks([])
        self.axs[0, 0].set_yticks([])

        for i in range(self.ndim):
            self.axs[i, i].set_yticklabels([])

        for ax in self.axs.flat:
            for loc in ["top", "right"]:
                ax.spines[loc].set_visible(False)

    def plot(
        self,
        x: np.ndarray,
        bins: int,
        limits: list[tuple[float, float]],
        diag_kws: dict = None,
        **plot_kws,
    ) -> None:

        if diag_kws is None:
            diag_kws = {}

        diag_kws.setdefault("color", "black")
        diag_kws.setdefault("lw", 1.25)
        diag_kws.setdefault("kind", "stairs")

        for i in range(self.ndim):
            for j in range(i):
                axis = (j, i)
                plot_points(
                    x[:, axis],
                    bins=bins,
                    limits=[limits[k] for k in axis],
                    ax=self.axs[i, j],
                    **plot_kws,
                )

        for i in range(self.ndim):
            plot_points_1d(
                x[:, i], bins=bins, limits=limits[i], ax=self.axs[i, i], **diag_kws
            )

        self.set_limits(limits)

    def set_limits(self, limits: list[tuple[float, float]]) -> None:
        for i in range(self.ndim):
            for j in range(self.ndim):
                if i != j:
                    self.axs[i, j].set_ylim(limits[i])
                self.axs[i, j].set_xlim(limits[j])

    def set_labels(self, labels: list[str]) -> None:
        for i in range(self.ndim):
            self.axs[-1, i].set_xlabel(labels[i])
            if i > 0:
                self.axs[i, 0].set_ylabel(labels[i])
        self.fig.align_ylabels()


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
            Signature: `plot_dist(x: np.ndarray)`.
        plot_proj: list[callable]
            Plots simulated vs. actual measurements.
            Signature: `plot_proj(projections_meas: list[Histogram], projections_pred: list[Histogram])`.
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
        projections_true = copy_histograms(model.projections)
        projections_pred = copy_histograms(model.diagnostics)

        projections_pred = simulate_with_diag_update(
            x_pred,
            model.transforms,
            projections_pred,
            kde=False,
            blur=False,
        )

        projections_true = unravel(projections_true)
        projections_pred = unravel(projections_pred)

        # Make plots
        figs = []

        ## Plot samples
        if self.plot_dist is not None:
            for function in self.plot_dist:
                fig, axs = function(x_pred)
                figs.append(fig)

        ## Plot measured vs. simulated projections
        if self.plot_proj is not None:
            for function in self.plot_proj:
                fig, axs = function(projections_pred, projections_true)
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
        projections_pred: list[Histogram1D],
        projections_true: list[Histogram1D],
    ):
        nmeas = len(projections_pred)
        ncols = min(self.ncols, nmeas)
        nrows = int(np.ceil(nmeas / ncols))

        self.fig_kws.setdefault("ncols", ncols)
        self.fig_kws.setdefault("nrows", nrows)
        self.fig_kws.setdefault("figsize", (1.50 * ncols, 1.1 * nrows))

        fig, axs = plt.subplots(**self.fig_kws)

        for i, (proj_pred, proj_true) in enumerate(
            zip(projections_pred, projections_true)
        ):
            ax = axs.flat[i]

            proj_pred = proj_pred.copy()
            proj_true = proj_true.copy()

            scale = proj_true.values.max()
            proj_pred.values /= scale
            proj_true.values /= scale

            plot_image_1d(
                proj_pred.values, proj_pred.edges, ax=ax, **self.plot_kws_pred
            )
            plot_image_1d(
                proj_true.values, proj_true.edges, ax=ax, **self.plot_kws_true
            )
            ax.set_ylim(self.ymin, self.ymax)
            ax.set_xlim(self.xmin, self.xmax)

        if self.xlim_scale is not None:
            for ax in axs:
                xlim = np.array(ax.get_xlim())
                xlim = xlim * self.xlim_scale
                ax.set_xlim(xlim)

        return (fig, axs)


class PlotProj2DContour:
    def __init__(
        self,
        ncols_max: int = 7,
        lim_share: bool = True,
        lim_scale: float = 1.0,
        plot_kws_true: dict = None,
        plot_kws_pred: dict = None,
        **plot_kws,
    ) -> None:
        self.ncols_max = ncols_max
        self.lim_share = lim_share
        self.lim_scale = lim_scale

        self.plot_kws = plot_kws

        self.plot_kws_true = plot_kws_true
        if self.plot_kws_true is None:
            self.plot_kws_true = {}

        self.plot_kws_pred = plot_kws_pred
        if self.plot_kws_pred is None:
            self.plot_kws_pred = {}

        self.plot_kws.setdefault("kind", "contour")
        self.plot_kws.setdefault("proc_kws", {})
        self.plot_kws["proc_kws"].setdefault("scale", "max")
        self.plot_kws["proc_kws"].setdefault("blur", 0.0)
        self.plot_kws.setdefault("levels", np.linspace(0.01, 1.0, 7))
        self.plot_kws.setdefault("lw", 0.75)

        self.plot_kws_true.setdefault("colors", "black")
        self.plot_kws_pred.setdefault("colors", "red")

        for key, val in self.plot_kws.items():
            self.plot_kws_true[key] = val
            self.plot_kws_pred[key] = val

    def __call__(
        self,
        projections_pred: list[HistogramND],
        projections_true: list[HistogramND],
    ) -> tuple:
        nmeas = len(projections_true)
        ncols = min(nmeas, self.ncols_max)
        nrows = int(np.ceil(nmeas / ncols))

        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(1.7 * ncols, 1.7 * nrows),
            sharex=self.lim_share,
            sharey=self.lim_share,
        )
        for index in range(nmeas):
            ax = axs.flat[index]
            proj_true = projections_true[index]
            proj_pred = projections_pred[index]
            plot_image(proj_true.values, proj_true.edges, ax=ax, **self.plot_kws_true)
            plot_image(proj_pred.values, proj_pred.edges, ax=ax, **self.plot_kws_pred)

        if self.lim_share:
            ax_ref = axs[0]
            for ax in axs.flat:
                ax.set_xlim(np.multiply(ax_ref.get_xlim(), self.lim_scale))
                ax.set_ylim(np.multiply(ax_ref.get_ylim(), self.lim_scale))
        else:
            for ax in axs.flat:
                ax.set_xlim(np.multiply(ax.get_xlim(), self.lim_scale))
                ax.set_ylim(np.multiply(ax.get_ylim(), self.lim_scale))

        return (fig, axs)


class PlotDistCorner:
    def __init__(
        self,
        bins: int,
        limits: list[tuple[float, float]],
        fig_kws: dict = None,
        diag_ymin: float = None,
        diag_ymax: float = None,
        plot_kws: dict = None,
    ) -> None:

        self.fig_kws = fig_kws
        if self.fig_kws is None:
            self.fig_kws = {}

        self.plot_kws = plot_kws
        if self.plot_kws is None:
            self.plot_kws = {}

        self.plot_kws.setdefault("bins", bins)
        self.plot_kws.setdefault("limits", limits)
        self.plot_kws.setdefault("cmap", "viridis")

        self.diag_ymin = diag_ymin
        self.diag_ymax = diag_ymax

    def __call__(self, x: np.ndarray) -> tuple:
        grid = CornerGrid(ndim=x.shape[1], **self.fig_kws)
        grid.plot(x, **self.plot_kws)
        return (grid.fig, grid.axs)
