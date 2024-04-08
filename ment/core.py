import math
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import scipy.interpolate

from .prior import GaussianPrior
from .prior import UniformPrior


class Histogram1D:
    def __init__(self, axis: int, bin_edges: np.ndarray) -> None:
        self.axis = axis
        self.ndim = 1
        self.bin_edges = bin_edges
        self.bin_coords = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.bin_volume = bin_edges[1] - bin_edges[0]
        
    def project(self, x: np.ndarray) -> np.ndarray:
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


class LagrangeFunction:
    def __init__(self, ndim: int, coords: List[np.array], values: np.array) -> None:
        self.ndim = ndim

        self.coords = coords
        if self.ndim == 1:
            self.coords = [self.coords]

        self.interpolator = None
        self.values = self.set_values(values)

    def set_values(self, values: np.ndarray) -> None:
        self.values = values
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            self.coords,
            self.values,
            method="linear",
            bounds_error=False, 
            fill_value=0.0,
        )
        return self.values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.interpolator(x)


class MENT:
    """Constrained maximum-entropy distribution."""
    def __init__(
        self, 
        ndim: int,
        measurements: List[List[np.ndarray]],
        transforms: List[Callable], 
        diagnostics: List[List[Any]],
        prior: Any,
        sampler: Callable,
        n_samples: int = 1_000_000, 
        verbose: bool = True,
    ) -> None:
        self.ndim = ndim
        self.verbose = verbose
        self.epoch = 0

        self.transforms = transforms
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(dim=dim, scale=100.0)

        self.lagrange_functions = self.initialize_lagrange_functions()
        
        self.sampler = sampler
        self.n_samples = int(n_samples)

    def set_diagnostics(self, diagnostics: List[List[Any]]) -> List[List[Any]]:
        self.diagnostics = diagnostics
        if self.diagnostics is None:
            self.diagnostics = [[]]
        return self.diagnostics

    def set_measurements(self, measurements: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = [[]]
        return self.measurements

    def initialize_lagrange_functions(self) -> List[List[np.ndarray]]:
        self.lagrange_functions = []
        for index in range(len(self.measurements)):
            self.lagrange_functions.append([])
            for measurement, diagnostic in zip(self.measurements[index], self.diagnostics[index]):
                values = (measurement > 0.0).astype(np.float32)
                lagrange_function = LagrangeFunction(
                    ndim=measurement.ndim, coords=diagnostic.bin_coords, values=values
                )
                self.lagrange_functions[-1].append(lagrange_function)
        return self.lagrange_functions

    def normalize_projection(self, projection: np.ndarray, index: int, diag_index: int) -> np.ndarray:
        diagnostic = self.diagnostics[index][diag_index]
        return projection / np.sum(projection) / diagnostic.bin_volume

    def evaluate_lagrange_function(self, u: np.ndarray, index: int, diag_index: int) -> np.ndarray:
        diagnostic = self.diagnostics[index][diag_index]
        lagrange_function = self.lagrange_functions[index][diag_index]
        return lagrange_function(diagnostic.project(u))

    def prob(self, x: np.ndarray) -> np.ndarray:
        prob = np.ones(x.shape[0])
        for index, transform in enumerate(self.transforms):
            u = transform(x)
            for diag_index, diagnostic in enumerate(self.diagnostics[index]):
                h = self.evaluate_lagrange_function(u, index, diag_index)
                # h = np.clip(h, 0.0, 1.00e+10)  # stability
                prob *= h
        prob *= self.prior.prob(x)
        return prob

    def sample(self, size: int) -> np.ndarray:
        x = self.sampler(self.prob, size)
        return x

    def simulate(self, index: int, diag_index: int) -> np.ndarray:
        x = self.sample(self.n_samples)
        transform = self.transforms[index]
        diagnostic = self.diagnostics[index][diag_index]
        prediction = diagnostic(transform(x))
        prediction = self.normalize_projection(prediction, index, diag_index)
        return prediction

    def gauss_seidel_step(self, lr: float = 1.0) -> None:
        for index, transform in enumerate(self.transforms):
            if self.verbose:
                print(f"index={index}")
            for diag_index, diagnostic in enumerate(self.diagnostics[index]):
                lagrange_function = self.lagrange_functions[index][diag_index]
                measurement = self.measurements[index][diag_index]
                prediction = self.simulate(index, diag_index)                
                shape = lagrange_function.values.shape
                lagrange_function.values = np.ravel(lagrange_function.values)
                for i, (g_meas, g_pred) in enumerate(zip(np.ravel(measurement), np.ravel(prediction))):
                    if (g_meas != 0.0) and (g_pred != 0.0):
                        lagrange_function.values[i] *= 1.0 + lr * ((g_meas / g_pred) - 1.0)
                lagrange_function.values = np.reshape(lagrange_function.values, shape)
                lagrange_function.set_values(lagrange_function.values)
                self.lagrange_functions[index][diag_index] = lagrange_function  # need this?
        self.epoch += 1

    def parameters(self):
        return

    def save(self, path) -> None:
        return 
        
    def load(self, path, device=None):
        return 



