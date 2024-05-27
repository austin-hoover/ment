import math
import time
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import scipy.interpolate

from .prior import GaussianPrior
from .prior import UniformPrior


class LagrangeFunction:
    def __init__(
        self, 
        ndim: int, 
        coords: List[np.array],
        values: np.array, 
        interpolation_kws: dict = None,
    ) -> None:
        self.ndim = ndim
        
        self.interpolation_kws = interpolation_kws
        if self.interpolation_kws is None:
            self.interpolation_kws = dict()
        self.interpolation_kws.setdefault("method", "linear")
        self.interpolation_kws.setdefault("bounds_error", False)
        self.interpolation_kws.setdefault("fill_value", 0.0)

        self.coords = coords
        if self.ndim == 1:
            self.coords = [self.coords]

        self.interpolator = None
        self.values = self.set_values(values)

    def set_values(self, values: np.ndarray) -> None:
        self.values = values
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            self.coords, self.values, **self.interpolation_kws
        )
        return self.values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.interpolator(x)


class MENT:
    def __init__(
        self, 
        ndim: int,
        measurements: List[List[np.ndarray]],
        transforms: List[Callable], 
        diagnostics: List[List[Any]],
        prior: Any,
        sampler: Callable,
        n_samples: int = 1_000_000, 
        integration_limits: List[Tuple[float]] = None,
        integration_size: int = None,
        integration_batches: int = None,
        interpolation: dict = None,
        verbose: bool = True,
        mode: str = "sample",
    ) -> None:
        self.ndim = ndim
        self.verbose = verbose
        self.epoch = 0
        self.mode = mode

        self.transforms = transforms
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(dim=dim, scale=100.0)

        self.lagrange_functions = self.initialize_lagrange_functions(**interpolation)
        
        self.sampler = sampler
        self.n_samples = int(n_samples)

        self.integration_limits = integration_limits
        self.integration_size = integration_size
        self.integration_batches = integration_batches

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

    def initialize_lagrange_functions(self, **kws) -> List[List[np.ndarray]]:
        self.lagrange_functions = []
        for index in range(len(self.measurements)):
            self.lagrange_functions.append([])
            for measurement, diagnostic in zip(self.measurements[index], self.diagnostics[index]):
                values = (measurement > 0.0).astype(np.float32)
                lagrange_function = LagrangeFunction(
                    ndim=measurement.ndim, 
                    coords=diagnostic.bin_coords, 
                    values=values, 
                    interpolation_kws=kws,
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
                prob *= self.evaluate_lagrange_function(u, index, diag_index)
        return prob * self.prior.prob(x)

    def sample(self, size: int) -> np.ndarray:
        x = self.sampler(self.prob, size)
        return x

    def get_meas_points(self, index: int, diag_index: int) -> np.ndarray:
        diag = self.diagnostics[index][diag_index]
        meas = self.measurements[index][diag_index]
        if meas.ndim == 1:
            return diag.bin_coords
        else:
            return np.vstack([C.ravel() for C in np.meshgrid(*diag.bin_coords, indexing="ij")]).T

    def get_integration_points(self, index: int, diag_index: int, method: str = "grid") -> np.ndarray:
        diag = self.diagnostics[index][diag_index]
        meas = self.measurements[index][diag_index]
        
        int_limits = self.integration_limits[index][diag_index]

        int_dim = self.ndim - meas.ndim
        int_res = int(self.integration_size ** (1.0 / int_dim))
        int_shape = tuple(int_dim * [int_res])

        meas_axis = diag.axis
        if type(meas_axis) is int:
            meas_axis = (meas_axis,)
            int_limits = [int_limits,]
        int_axis = tuple([axis for axis in range(self.ndim) if axis not in meas_axis])

        if method == "grid":
            int_coords = [
                np.linspace(int_limits[i][0], int_limits[i][1], int_shape[i]) 
                for i in range(len(int_axis))
            ]

            int_points = None
            if len(int_axis) == 1:
                int_coords = int_coords[0]
                int_points = int_coords
            else:
                int_points = np.vstack([C.ravel() for C in np.meshgrid(*int_coords, indexing="ij")]).T
        elif method == "uniform":
            raise NotImplementedError
        elif method == "gaussian":
            raise NotImplementedError
        else:
            raise ValueError("Invalid method.")

        return int_points
    
    def simulate(self, index: int, diag_index: int) -> np.ndarray:
        transform = self.transforms[index]
        diag = self.diagnostics[index][diag_index]
        meas = self.measurements[index][diag_index]
        meas_axis = diag.axis
        integration_axis = [i for i in range(self.ndim) if i != meas_axis]
        pred = np.zeros(meas.shape)            
        
        if self.mode == "sample":
            x = self.sample(self.n_samples)
            u = transform(x)
            pred = diag(u)

        elif self.mode == "integrate":
            ## Pixel by pixel...
            diagnostic = self.diagnostics[index][diag_index]
            measurement = self.measurements[index][diag_index]
            transform = self.transforms[index]
    
            # Define measurement axis.
            meas_axis = diagnostic.axis
            if type(meas_axis) is int:
                meas_axis = (meas_axis,)
    
            # Define integration axis.
            int_axis = tuple([axis for axis in range(self.ndim) if axis not in meas_axis])
    
            # Get measurement and integration points.
            meas_points = self.get_meas_points(index, diag_index)
            int_points = self.get_integration_points(index, diag_index)
    
            # Initialize transformed coordinate array.
            u = np.zeros((int_points.shape[0], self.ndim))
            for k, axis in enumerate(int_axis):
                if len(int_axis) == 1:
                    u[:, axis] = int_points
                else:
                    u[:, axis] = int_points[:, k]
    
            # Compute integral.
            prediction = np.zeros(meas_points.shape[0])
            for i, meas_point in enumerate(meas_points):
                # Update the coordinates in the measurement plane.
                for k, axis in enumerate(meas_axis):
                    if measurement.ndim == 1:
                        u[:, axis] = meas_point
                    else:
                        u[:, axis] = meas_point[k]
                # Compute the probability density at the integration points.
                x = transform.inverse(u)
                prob = self.prob(x)  # assume symplectic transform
                # Integrate (ignore scaling factor)
                prediction[i] = np.sum(prob)
    
            # Reshape the flattened projection.
            if measurement.ndim > 1:
                prediction = np.reshape(prediction, measurement.shape)
    
            # Normalize the projection.
            prediction = self.normalize_projection(prediction, index, diag_index)
            return prediction
        

        elif self.mode == "integrate_batched":    
            # Experimental batched version...
            integration_batch_size = int(self.integration_size / self.integration_batches)
            u = np.zeros((integration_batch_size * meas.size, self.ndim))

            if meas.ndim == 1:
                meas_points = diag.bin_coords
                integration_axis = [i for i in range(self.ndim) if i != meas_axis][0]
    
                u[:, meas_axis] = np.repeat(meas_points, integration_batch_size)
                for _ in range(self.integration_batches):
                    lb = self.integration_limits[0]
                    ub = self.integration_limits[1]
                    u[:, integration_axis] = np.random.uniform(lb, ub, size=u.shape[0])
                    x = transform.inverse(u)
                    prob = self.prob(x)
                    prob = np.array(np.split(prob, meas.size))
                    pred += np.sum(prob, axis=1)
            else:
                meas_points = np.vstack([C.ravel() for C in np.meshgrid(*diag.bin_coords, indexing="ij")]).T
                u[:, meas_axis] = np.repeat(meas_points, integration_batch_size, axis=0)
                for _ in range(self.integration_batches):
                    lb = [xmin for (xmin, xmax) in self.integration_limits]
                    ub = [xmax for (xmin, xmax) in self.integration_limits]
                    u[:, integration_axis] = np.random.uniform(lb, ub, size=(u.shape[0], len(integration_axis)))
                    x = transform.inverse(u)
                    prob = model.prob(x)
                    prob = np.array(np.split(prob, meas.size))
                    pred += np.reshape(np.sum(prob, axis=1), meas.shape)  # error here?

        pred = self.normalize_projection(pred, index, diag_index)
        return pred


    def gauss_seidel_step(self, lr: float = 1.0, **kws) -> None:
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
                self.lagrange_functions[index][diag_index] = lagrange_function
        self.epoch += 1

    def parameters(self):
        return

    def save(self, path) -> None:
        return 
        
    def load(self, path, device=None):
        return 




class MENTRing:
    def __init__(
        self, 
        ndim: int,
        measurements: list[list[np.ndarray]],
        diagnostics: list[list[Any]],
        turn_indices: list[int],
        transform: Callable,
        prior: Any,
        sampler: Callable,
        n_samples: int = 1_000_000, 
        integration_limits: list[tuple[float]] = None,
        integration_size: int = None,
        integration_batches: int = None,
        interpolation: dict = None,
        verbose: bool = True,
        mode: str = "sample",
    ) -> None:
        self.ndim = ndim
        self.verbose = verbose
        self.epoch = 0
        self.mode = mode

        self.transform = transform
        self.turn_indices = turn_indices
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(dim=dim, scale=100.0)

        self.lagrange_functions = self.initialize_lagrange_functions(**interpolation)
        
        self.sampler = sampler
        self.n_samples = int(n_samples)

        self.integration_limits = integration_limits
        self.integration_size = integration_size
        self.integration_batches = integration_batches
    
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

    def initialize_lagrange_functions(self, **kws) -> list[list[np.ndarray]]:
        self.lagrange_functions = []
        for index in range(len(self.measurements)):
            self.lagrange_functions.append([])
            for measurement, diagnostic in zip(self.measurements[index], self.diagnostics[index]):
                values = (measurement > 0.0).astype(np.float32)
                lagrange_function = LagrangeFunction(
                    ndim=measurement.ndim, 
                    coords=diagnostic.bin_coords, 
                    values=values, 
                    interpolation_kws=kws,
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
        
        u = np.copy(x)
        index = 0
        for turn in range(max(self.turn_indices)):
            u = self.transform(u)
            if turn + 1 in self.turn_indices:
                for diag_index, diagnostic in enumerate(self.diagnostics[index]):
                    prob *= self.evaluate_lagrange_function(u, index, diag_index)
                index += 1
                
        return prob * self.prior.prob(x)

    def sample(self, size: int) -> np.ndarray:
        x = self.sampler(self.prob, size)
        return x

    def get_meas_points(self, index: int, diag_index: int) -> np.ndarray:
        diag = self.diagnostics[index][diag_index]
        meas = self.measurements[index][diag_index]
        if meas.ndim == 1:
            return diag.bin_coords
        else:
            return np.vstack([C.ravel() for C in np.meshgrid(*diag.bin_coords, indexing="ij")]).T

    def get_integration_points(self, index: int, diag_index: int, method: str = "grid") -> np.ndarray:
        diag = self.diagnostics[index][diag_index]
        meas = self.measurements[index][diag_index]
        
        int_limits = self.integration_limits[index][diag_index]

        int_dim = self.ndim - meas.ndim
        int_res = int(self.integration_size ** (1.0 / int_dim))
        int_shape = tuple(int_dim * [int_res])

        meas_axis = diag.axis
        if type(meas_axis) is int:
            meas_axis = (meas_axis,)
            int_limits = [int_limits,]
        int_axis = tuple([axis for axis in range(self.ndim) if axis not in meas_axis])

        if method == "grid":
            int_coords = [
                np.linspace(int_limits[i][0], int_limits[i][1], int_shape[i]) 
                for i in range(len(int_axis))
            ]

            int_points = None
            if len(int_axis) == 1:
                int_coords = int_coords[0]
                int_points = int_coords
            else:
                int_points = np.vstack([C.ravel() for C in np.meshgrid(*int_coords, indexing="ij")]).T
        elif method == "uniform":
            raise NotImplementedError
        elif method == "gaussian":
            raise NotImplementedError
        else:
            raise ValueError("Invalid method.")

        return int_points

    def simulate(self, index: int, diag_index: int) -> np.ndarray:
        diag = self.diagnostics[index][diag_index]
        meas = self.measurements[index][diag_index]
        meas_axis = diag.axis
        integration_axis = [i for i in range(self.ndim) if i != meas_axis]
                
        x = self.sample(self.n_samples)
        for turn in range(max(self.turn_indices)):
            x = self.transform(x)
            if turn + 1 == self.turn_indices[index]:
                pred = diag(x)
                pred = self.normalize_projection(pred, index, diag_index)
                return pred

    def gauss_seidel_step(self, lr: float = 1.0, **kws) -> None:
        for index in range(len(self.turn_indices)):
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
                self.lagrange_functions[index][diag_index] = lagrange_function
        self.epoch += 1

    def parameters(self):
        return

    def save(self, path) -> None:
        return 
        
    def load(self, path, device=None):
        return 
