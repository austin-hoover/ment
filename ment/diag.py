import math
from typing import List
from typing import Tuple

import numpy as np


class Histogram1D:
    def __init__(self, axis: int, bin_edges: np.ndarray, direction: np.ndarray = None) -> None:
        self.axis = axis
        self.direction = direction
        self.ndim = 1
        self.bin_edges = bin_edges
        self.bin_coords = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.bin_volume = bin_edges[1] - bin_edges[0]
        
    def project(self, x: np.ndarray) -> np.ndarray:
        if self.direction is not None:
            return np.sum(x * self.direction, axis=1)
        return x[:, self.axis]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(self.project(x), self.bin_edges, density=True)
        return hist


class HistogramND:
    def __init__(self, axis: Tuple[int], bin_edges: List[np.ndarray]) -> None:
        self.axis = axis
        self.ndim = len(axis)
        self.bin_edges = bin_edges
        self.bin_coords = [0.5 * (e[:-1] + e[1:]) for e in bin_edges]
        self.bin_volume = math.prod((e[1] - e[0]) for e in bin_edges)

    def project(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.axis]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hist, _ = np.histogramdd(self.project(x), self.bin_edges, density=True)
        return hist