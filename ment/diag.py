import copy
import math
from typing import Union
from typing import Self

import numpy as np
import scipy.ndimage
import scipy.stats

from .grid import coords_to_edges
from .grid import edges_to_coords
from .grid import get_grid_points


class HistogramND:
    def __init__(
        self,
        axis: tuple[int, ...],
        edges: list[np.ndarray] = None,
        coords: list[np.ndarray] = None,
        values: np.ndarray = None,
        kde: bool = False,
        kde_bandwidth_frac: float = 1.0,
        blur: float = 0.0,
        thresh: float = 0.0,
        thresh_type: str = "abs",
        store_grid_points: bool = True,
    ) -> None:
        self.axis = axis
        self.ndim = len(axis)

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = [edges_to_coords(e) for e in self.edges]
        if self.edges is None and self.coords is not None:
            self.edges = [coords_to_edges(c) for c in self.coords]

        self.shape = tuple([len(c) for c in self.coords])
        self.values = values
        if self.values is None:
            self.values = np.zeros(self.shape)

        self.bin_sizes = [
            self.coords[i][1] - self.coords[i][0] for i in range(self.ndim)
        ]
        self.bin_volume = np.prod(self.bin_sizes)
        self.grid_shape = tuple([len(c) for c in self.coords])
        self.grid_points = None
        self.store_grid_points = store_grid_points

        self.kde = kde
        self.kde_bandwidth_frac = kde_bandwidth_frac
        self.kde_bandwidth = kde_bandwidth_frac * np.mean(self.bin_sizes)

        self.blur = blur
        self.thresh = thresh
        self.thresh_type = thresh_type

    def sample(self, size: int, noise: float = 0.0) -> np.ndarray:
        return _sample_grid(
            values=self.values, edges=self.edges, size=size, noise=noise
        )

    def cov(self) -> np.ndarray:
        if self.ndim > 2:
            raise NotImplementedError

        return _get_hist_cov_2d(self)

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def normalize(self) -> None:
        values = np.copy(self.values)
        values_sum = np.sum(values)
        if values_sum > 0.0:
            values = values / values_sum / self.bin_volume
        self.values = values

    def get_grid_points(self) -> np.ndarray:
        if self.grid_points is not None:
            return self.grid_points

        grid_points = get_grid_points(self.coords)

        if self.store_grid_points:
            self.grid_points = grid_points

        return grid_points

    def project(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.axis]

    def bin(self, x: np.ndarray) -> np.ndarray:
        x_proj = self.project(x)

        values = None
        if self.kde:
            estimator = scipy.stats.gaussian_kde(x_proj.T, bw_method=self.kde_bandwidth)
            grid_points = self.get_grid_points()
            values = estimator(grid_points.T).reshape(self.grid_shape)
        else:
            values, _ = np.histogramdd(x_proj, bins=self.edges, density=True)

        if self.blur:
            values = scipy.ndimage.gaussian_filter(values, self.blur)

        if self.thresh:
            max_value = self.thresh
            if self.thresh_type == "frac":
                max_value = max_value * np.max(values)
            values[values < max_value] = 0.0

        self.values = values
        self.normalize()
        return self.values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.bin(x)


class Histogram1D:
    def __init__(
        self,
        axis: int,
        edges: np.ndarray = None,
        coords: np.ndarray = None,
        values: np.ndarray = None,
        direction: np.ndarray = None,
        kde: bool = False,
        kde_bandwidth_frac: float = 1.0,
        blur: float = 0.0,
        thresh: float = 0.0,
        thresh_type: str = "abs",
    ) -> None:
        self.axis = axis
        self.ndim = 1

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = edges_to_coords(self.edges)
        if self.edges is None and self.coords is not None:
            self.edges = coords_to_edges(self.coords)

        self.shape = len(self.coords)
        self.values = values
        if self.values is None:
            self.values = np.zeros(self.shape)

        self.direction = direction
        self.bin_size = self.coords[1] - self.coords[0]
        self.bin_volume = self.bin_width = self.bin_size

        self.kde = kde
        self.kde_bandwidth_frac = kde_bandwidth_frac
        self.kde_bandwidth = kde_bandwidth_frac * self.bin_size
        self.blur = blur
        self.thresh = thresh
        self.thresh_type = thresh_type

    def sample(self, size: int, noise: float = 0.0) -> np.ndarray:
        return _sample_grid(
            values=self.values, edges=self.edges, size=size, noise=noise
        )

    def var(self) -> float:
        values_sum = np.sum(self.values)
        if values_sum <= 0.0:
            raise ValueError("Histogram values are zero.")

        x = np.copy(self.coords)
        f = np.copy(self.values)
        x_avg = np.average(x, weights=f)
        x_var = np.average((x - x_avg) ** 2, weights=f)
        return x_var

    def std(self) -> float:
        return np.sqrt(self.var())

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def get_grid_points(self) -> np.ndarray:
        return self.coords

    def normalize(self) -> None:
        values = np.copy(self.values)
        values_sum = np.sum(values)
        if values_sum > 0.0:
            values = values / values_sum / self.bin_volume
        self.values = values

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.direction is not None:
            return np.sum(x * self.direction, axis=1)
        return x[:, self.axis]

    def bin(self, x: np.ndarray) -> np.ndarray:
        x_proj = self.project(x)

        values = None
        if self.kde:
            estimator = scipy.stats.gaussian_kde(x_proj, bw_method=self.kde_bandwidth)
            values = estimator(self.coords)
        else:
            values, _ = np.histogram(x_proj, self.edges, density=True)

        if self.blur:
            values = scipy.ndimage.gaussian_filter(values, self.blur)

        if self.thresh:
            max_value = self.thresh
            if self.thresh_type == "frac":
                max_value = max_value * np.max(values)
            values[values < max_value] = 0.0

        self.values = values
        self.normalize()
        return self.values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.bin(x)


def _sample_grid(
    values: np.ndarray, edges: list[np.ndarray], size: int = 100, noise: float = 0.0
) -> np.ndarray:
    if np.ndim(edges) == 1:
        edges = [edges]

    idx = np.flatnonzero(values)
    pdf = values.ravel()[idx]
    pdf = pdf / np.sum(pdf)
    idx = np.random.choice(idx, size, replace=True, p=pdf)
    idx = np.unravel_index(idx, shape=values.shape)
    lb = [edges[axis][idx[axis]] for axis in range(values.ndim)]
    ub = [edges[axis][idx[axis] + 1] for axis in range(values.ndim)]

    points = np.squeeze(np.random.uniform(lb, ub).T)
    if noise:
        for axis in range(points.shape[1]):
            delta = ub[axis] - lb[axis]
            points[:, axis] += (
                noise * 0.5 * np.random.uniform(-delta, delta, size=points.shape[0])
            )
    return points


def _get_hist_cov_2d(hist: HistogramND) -> np.ndarray:
    values = hist.values
    coords = hist.coords
    ndim = values.ndim

    S = np.zeros((ndim, ndim))

    values_sum = np.sum(values)
    if values_sum <= 0.0:
        return S

    COORDS = np.meshgrid(*coords, indexing="ij")
    coords_mean = np.array([np.average(C, weights=values) for C in COORDS])
    for i in range(ndim):
        for j in range(i + 1):
            X = COORDS[i] - coords_mean[i]
            Y = COORDS[j] - coords_mean[j]
            EX = np.sum(values * X) / values_sum
            EY = np.sum(values * Y) / values_sum
            EXY = np.sum(values * X * Y) / values_sum
            S[i, j] = S[j, i] = EXY - EX * EY
    return S
