import math
import time
import pickle
from typing import Any
from typing import Callable
from typing import Union

import numpy as np
import scipy.interpolate
from tqdm import tqdm

from .diag import Histogram1D
from .diag import HistogramND
from .prior import GaussianPrior
from .prior import UniformPrior
from .grid import coords_to_edges
from .grid import edges_to_coords
from .grid import get_grid_points
from .utils import wrap_tqdm


class LagrangeFunction:
    def __init__(
        self, 
        ndim: int, 
        coords: list[np.array],
        values: np.array, 
        **interp_kws,
    ) -> None:
        self.ndim = ndim
        
        self.interp_kws = interp_kws
        self.interp_kws.setdefault("method", "linear")
        self.interp_kws.setdefault("bounds_error", False)
        self.interp_kws.setdefault("fill_value", 0.0)

        self.coords = coords
        if self.ndim == 1:
            self.coords = [self.coords]

        self.interp = None
        self.values = self.set_values(values)

    def set_values(self, values: np.ndarray) -> None:
        self.values = values
        self.interp = scipy.interpolate.RegularGridInterpolator(self.coords, self.values, **self.interp_kws)
        return self.values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.interp(x)


class MENT:
    def __init__(
        self, 
        ndim: int,
        transforms: list[Callable], 
        diagnostics: list[list[Any]],
        measurements: list[list[np.ndarray]],
        prior: Any,
        sampler: Callable,
        n_samples: int = 1_000_000, 
        integration_limits: list[tuple[float, float]] = None,
        integration_size: int = None,
        integration_batches: int = None,
        interpolation_kws: dict = None,
        verbose: Union[bool, int] = True,
        mode: str = "sample",
    ) -> None:
        
        self.ndim = ndim
        self.verbose = int(verbose)
        self.mode = mode

        self.transforms = transforms
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(ndim, scale=100.0)

        if interpolation_kws is None:
            interpolation_kws = dict()
        self.lagrange_functions = self.init_lagrange_functions(**interpolation_kws)
        
        self.sampler = sampler
        self.n_samples = int(n_samples)

        self.integration_limits = integration_limits
        self.integration_size = integration_size
        self.integration_batches = integration_batches

        self.epoch = 0

    def set_diagnostics(self, diagnostics: list[list[Any]]) -> list[list[Any]]:
        self.diagnostics = diagnostics
        if self.diagnostics is None:
            self.diagnostics = [[]]
        return self.diagnostics

    def set_measurements(self, measurements: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = [[]]
        return self.measurements

    def init_lagrange_functions(self, **interp_kws) -> list[list[np.ndarray]]:
        self.lagrange_functions = []
        for index in range(len(self.measurements)):
            self.lagrange_functions.append([])
            for measurement, diagnostic in zip(self.measurements[index], self.diagnostics[index]):
                values = measurement > 0.0
                values = values.astype(float)
                coords = diagnostic.coords
                lagrange_function = LagrangeFunction(
                    ndim=values.ndim, 
                    coords=coords,
                    values=values, 
                    **interp_kws,
                )
                self.lagrange_functions[-1].append(lagrange_function)
        return self.lagrange_functions

    def prob(self, x: np.ndarray) -> np.ndarray:
        prob = np.ones(x.shape[0])
        for index, transform in enumerate(self.transforms):
            u = transform(x)
            for diagnostic, lagrange_function in zip(self.diagnostics[index], self.lagrange_functions[index]):
                prob *= lagrange_function(diagnostic.project(u))
        return prob * self.prior.prob(x)

    def sample(self, size: int) -> np.ndarray:
        return self.sampler(self.prob, size)

    def get_measurement_points(self, index: int, diag_index: int) -> np.ndarray:
        diagnostic = self.diagnostics[index][diag_index]
        return diagnostic.get_grid_points()

    def get_integration_points(self, index: int, diag_index: int, method: str = "grid") -> np.ndarray:
        measurement = self.measurements[index][diag_index]
        diagnostic = self.diagnostics[index][diag_index]

        measurement_axis = diagnostic.axis
        if type(measurement_axis) is int:
            measurement_axis = (measurement_axis,)
            
        integration_axis = tuple([axis for axis in range(self.ndim) if axis not in measurement_axis])
        integration_ndim = len(integration_axis)
        integration_limits = self.integration_limits[index][diag_index]
        integration_size = self.integration_size
        integration_points = None

        if (integration_ndim == 1) and (np.ndim(integration_limits) == 1):
            integration_limits = [integration_limits]
        
        if method == "grid":
            integration_grid_resolution = int(integration_size ** (1.0 / integration_ndim))
            integration_grid_shape = tuple(integration_ndim * [integration_grid_resolution])
            integration_grid_coords = [
                np.linspace(integration_limits[i][0], integration_limits[i][1], integration_grid_shape[i]) 
                for i in range(integration_ndim)
            ]
            if integration_ndim == 1:
                integration_points = integration_grid_coords[0]
            else:
                integration_points = get_grid_points(integration_grid_coords)
        else:
            raise NotImplementedError
            
        return integration_points
    
    def simulate(self, index: int, diag_index: int) -> np.ndarray:
        transform = self.transforms[index]
        diagnostic = self.diagnostics[index][diag_index]
        values_meas = self.measurements[index][diag_index]
        values_pred = np.zeros(values_meas.shape)          

        if self.mode == "sample":
            x = self.sample(self.n_samples)
            return diagnostic(transform(x))

        measurement_axis = diagnostic.axis
        if type(measurement_axis) is int:
            measurement_axis = (measurement_axis,)
        measurement_ndim = len(measurement_axis)

        integration_axis = [axis for axis in range(self.ndim) if axis not in measurement_axis]
        integration_axis = tuple(integration_axis)
        integration_ndim = len(integration_axis)
        integration_limits = self.integration_limits[index][diag_index]
        
        measurement_points = self.get_measurement_points(index, diag_index)
        integration_points = self.get_integration_points(index, diag_index)

        if self.mode == "integrate":         
            u = np.zeros((integration_points.shape[0], self.ndim))
            for k, axis in enumerate(integration_axis):
                if integration_ndim == 1:
                    u[:, axis] = integration_points
                else:
                    u[:, axis] = integration_points[:, k]
    
            values_pred = np.zeros(measurement_points.shape[0])            
            for i, point in enumerate(wrap_tqdm(measurement_points, self.verbose > 1)):
                for k, axis in enumerate(measurement_axis):
                    if values_meas.ndim == 1:
                        u[:, axis] = point
                    else:
                        u[:, axis] = point[k]
                prob = self.prob(transform.inverse(u))  # symplectic
                values_pred[i] = np.sum(prob)
    
            if values_meas.ndim > 1:
                values_pred = np.reshape(values_pred, values_meas.shape)
    
        elif self.mode == "integrate_batched":    
            # Experimental batched version...
            integration_batch_size = int(self.integration_size / self.integration_batches)
            u = np.zeros((integration_batch_size * values_meas.size, self.ndim))
            
            if measurement_ndim == 1:
                measurement_axis = measurement_axis[0]
                u[:, measurement_axis] = np.repeat(measurement_points, integration_batch_size)
                for _ in range(self.integration_batches):
                    lb = integration_limits[0]
                    ub = integration_limits[1]
                    u[:, integration_axis] = np.random.uniform(lb, ub, size=(u.shape[0], integration_ndim))
                    prob = self.prob(transform.inverse(u))
                    prob = np.array(np.split(prob, values_meas.size))
                    values_pred += np.sum(prob, axis=1)
            else:
                u[:, measurement_axis] = np.repeat(measurement_points, integration_batch_size, axis=0)
                for _ in range(self.integration_batches):
                    lb = [xmin for (xmin, xmax) in integration_limits]
                    ub = [xmax for (xmin, xmax) in integration_limits]
                    u[:, integration_axis] = np.random.uniform(lb, ub, size=(u.shape[0], integration_ndim))
                    prob = self.prob(transform.inverse(u))
                    prob = np.array(np.split(prob, values_meas.size))
                    values_pred += np.reshape(np.sum(prob, axis=1), values_meas.shape)  # error?

        else:
            raise ValueError
                    
        values_pred = diagnostic.normalize(values_pred)
        return values_pred

    def gauss_seidel_step(self, learning_rate: float = 1.0, **kws) -> None:
        for index, transform in enumerate(self.transforms):
            if self.verbose:
                print(f"index={index}")
            for diag_index, diagnostic in enumerate(self.diagnostics[index]):
                lagrange_function = self.lagrange_functions[index][diag_index]
                values_meas = self.measurements[index][diag_index]
                values_pred = self.simulate(index, diag_index)    
                
                shape = lagrange_function.values.shape
                lagrange_function.values = np.ravel(lagrange_function.values)
                for i, (val_meas, val_pred) in enumerate(zip(np.ravel(values_meas), np.ravel(values_pred))):
                    if (val_meas != 0.0) and (val_pred != 0.0):
                        lagrange_function.values[i] *= 1.0 + learning_rate * ((val_meas / val_pred) - 1.0)
                lagrange_function.values = np.reshape(lagrange_function.values, shape)
                lagrange_function.set_values(lagrange_function.values)
                self.lagrange_functions[index][diag_index] = lagrange_function
        self.epoch += 1

    def parameters(self):
        return

    def save(self, path: str) -> None:
        state = {
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "measurements": self.measurements,

            "ndim": self.ndim,
            "prior": self.prior,
            "sampler": self.sampler,    

            "epoch": self.epoch,
            "lagrange_functions": self.lagrange_functions,
        }
        
        file = open(path, "wb")
        pickle.dump(state, file, pickle.HIGHEST_PROTOCOL)
        file.close()
        
    def load(self, path: str) -> None:
        file = open(path, "rb")
        
        state = pickle.load(file)
        
        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.measurements = state["measurements"]

        self.ndim = state["ndim"]
        self.prior = state["prior"]
        self.sampler = state["sampler"]    

        self.epoch = state["epoch"]
        self.lagrange_functions = state["lagrange_functions"]
        
        file.close()

