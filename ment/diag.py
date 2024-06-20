import numpy as np

from .grid import coords_to_edges
from .grid import edges_to_coords
from .grid import get_grid_points


class Histogram1D:
    def __init__(self, axis: int, edges: np.ndarray, direction: np.ndarray = None) -> None:
        self.axis = axis
        self.ndim = 1
        self.edges = edges
        self.coords = edges_to_coords(edges)
        self.direction = direction
        self.grid_points = None

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return values / np.sum(values) / (self.edges[1] - self.edges[0])
        
    def project(self, x: np.ndarray) -> np.ndarray:
        if self.direction is not None:
            return np.sum(x * self.direction, axis=1)
        return x[:, self.axis]

    def get_grid_points(self) -> np.ndarray:
        self.grid_points = np.copy(self.coords)
        return self.grid_points

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(self.project(x), self.edges, density=True)
        return hist


class HistogramND:
    def __init__(self, axis: tuple[int, ...], edges: list[np.ndarray], store_grid_points: bool = True) -> None:
        self.axis = axis
        self.ndim = len(axis)
        self.edges = edges
        self.coords = [0.5 * (e[:-1] + e[1:]) for e in edges]
        self.grid_points = None
        self.store_grid_points = store_grid_points

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return values / np.sum(values) / np.prod([e[1] - e[0] for e in self.edges])

    def project(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.axis]

    def get_grid_points(self) -> np.ndarray:
        if self.grid_points is not None:
            return self.grid_points
        grid_points = get_grid_points(self.coords)
        if self.store_grid_points:
            self.grid_points = grid_points
        return grid_points
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        hist, _ = np.histogramdd(self.project(x), self.edges, density=True)
        return hist


