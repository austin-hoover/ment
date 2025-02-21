import math
import time
import pickle
from typing import Any
from typing import Callable
from typing import TypeAlias
from typing import Union

import numpy as np
import psdist as ps
import scipy.interpolate
from tqdm import tqdm

from .diag import Histogram1D
from .diag import HistogramND
from .grid import get_grid_points
from .prior import UniformPrior
from .prior import GaussianPrior
from .sim import IdentityTransform
from .sim import LinearTransform
from .utils import wrap_tqdm
from .utils import unravel


Histogram: TypeAlias = Union[Histogram1D, HistogramND]  # python<3.12 compatible


class LagrangeFunction:
    """Represents Lagrange multiplier function on regular grid.

    This function can be evaluated at any point by interpolation.
    """

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
        self.interp = scipy.interpolate.RegularGridInterpolator(
            self.coords, self.values, **self.interp_kws
        )
        return self.values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.interp(x)


class MENT:
    """Maximum-Entropy Tomography (MENT) model.

    NOTE: MENT reconstructs the distribution in normalized coordinates z. The normalized
    coordinates are related to the real coordinates by a linear transformation
    x = Vz. To generate samples from the real distribution, call `unnormalize`:

    ```
    model = MENT(...)
    z = model.sample(100_000)
    x = model.unnormalize(z)
    ```

    Similarly, to compute the probability density at x:

    ```
    prob = model.prob(model.normalize(x))
    ```

    There is no need to do this if V = I = identity matrix.
    """

    def __init__(
        self,
        ndim: int,
        transforms: list[Callable],
        projections: list[list[Histogram]],
        prior: Any,
        sampler: Callable,
        unnorm_matrix: np.ndarray = None,
        nsamp: int = 1_000_000,
        integration_limits: list[tuple[float, float]] = None,
        integration_size: int = None,
        store_integration_points: bool = True,
        interpolation_kws: dict = None,
        verbose: Union[bool, int] = True,
        mode: str = "sample",
    ) -> None:
        """Constructor.

        Parameters
        ----------
        ndim : int
            Number of phase space dimensions.
        transforms : list[Callable]
            A list of functions that transform the phase space coordinates. Call
            signature is `transform(x: np.ndarray) -> np.ndarray`.
        projections : list[list[Histogram]]
            List of measured projections, which we store as HistogramND or Histogram1D
            objects. We provide a list of projections for each transform.
        unnorm_matrix : np.ndarray
            Matrix V that **unnormalizes** the phase space coordinates: x = Vz.
            Defaults to identity matrix. If V = I, ignore all comments about normalized
            and unnormalized coordinates.
        prior : Any
            Prior distribution over the **normalized** phase space coordinates z = V^-1 x.
            Must implement `prior.prob(z: np.ndarray) -> np.ndarray`.
        sampler : Callable
            Calling `sampler(f, n)` generates n samples from the function f.
        nsamp : int
            Number of samples to use when computing projections. Only relevant if
            `self.mode=="sample".
        integration_limits : list[tuple[float, float]]
            List of (min, max) coordinates of integration grid.
        integration_size : int
            Number of integration points.
        store_integration_points : bool
            Whether to keep the integration points in memory.
        interpolation_kws : dict
            Key word arguments passed to `scipy.interpolate.RegularGridInterpolator` for
            interpolating the Lagrange multiplier functions.
        verbose : int
            Whether to print updates during calculations.
        mode : {"sample" or "forward", "integration" or "backward"}
            Whether to use numerical integration or particle sampling to compuate
            projections.
        """
        self.ndim = ndim
        self.verbose = int(verbose)
        self.mode = mode

        self.transforms = transforms
        self.projections = self.set_projections(projections)

        # Copy projection histograms for simulation.
        self.diagnostics = []
        for index in range(len(self.projections)):
            self.diagnostics.append([hist.copy() for hist in self.projections[index]])

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(ndim, scale=100.0)

        self.unnorm_matrix = unnorm_matrix
        self.unnorm_transform = self.set_unnorm_transform(unnorm_matrix)

        if interpolation_kws is None:
            interpolation_kws = dict()
        self.lagrange_functions = self.init_lagrange_functions(**interpolation_kws)

        self.sampler = sampler
        self.nsamp = int(nsamp)

        self.integration_limits = integration_limits
        self.integration_size = integration_size
        self.integration_points = None
        self.store_integration_points = store_integration_points

        self.epoch = 0

    def set_unnorm_transform(self, unnorm_matrix: np.ndarray) -> Callable:
        """Set inverse of normalization matrix.

        The unnormalization matrix transforms normalized coordinates z to
        phase space coordinates x via the linear mapping: x = Vz.
        """
        self.unnorm_matrix = unnorm_matrix
        if self.unnorm_matrix is None:
            self.unnorm_transform = IdentityTransform()
            self.unnorm_matrix = np.eye(self.ndim)
        else:
            self.unnorm_transform = LinearTransform(self.unnorm_matrix)
        return self.unnorm_transform

    def set_interpolation_kws(self, **kws) -> None:
        """Set interpolation key word arguments for lagrange functions.

        These arguments are passed to `scipy.interpolate.RegularGridInterpolator`.
        """
        kws.setdefault("method", "linear")
        kws.setdefault("bounds_error", False)
        kws.setdefault("fill_value", 0.0)

        for i in range(len(self.lagrange_functions)):
            for j in range(len(self.lagrange_functions[i])):
                self.lagrange_functions[i][j].interp_kws = kws
                self.lagrange_functions[i][j].set_values(self.lagrange_functions[i][j].values)

    def set_projections(self, projections: list[list[Histogram]]) -> list[list[Histogram]]:
        """Set list of measured projections (histograms)."""
        self.projections = projections
        if self.projections is None:
            self.projections = [[]]
        return self.projections

    def init_lagrange_functions(self, **interp_kws) -> list[list[np.ndarray]]:
        """Initialize lagrange multipler functions.

        The function l(u_proj) = 1 if the measured projection g(u_proj) > 0,
        otherwise l(u_proj) = 0.

        Key word arguments passed to `LagrangeFunction` constructor.
        """
        self.lagrange_functions = []
        for index in range(len(self.projections)):
            self.lagrange_functions.append([])
            for projection in self.projections[index]:
                values = projection.values > 0.0
                values = values.astype(float)
                coords = projection.coords
                lagrange_function = LagrangeFunction(
                    ndim=values.ndim,
                    coords=coords,
                    values=values,
                    **interp_kws,
                )
                self.lagrange_functions[-1].append(lagrange_function)
        return self.lagrange_functions

    def unnormalize(self, z: np.ndarray) -> np.ndarray:
        """Unnormalize coordinates z: x = Vz."""
        if self.unnorm_transform is None:
            self.unnorm_transform = IdentityTransform()
        return self.unnorm_transform(z)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize coordinates x: z = V^-1 z."""
        return self.unnorm_transform.inverse(x)

    def prob(self, z: np.ndarray, squeeze: bool = True) -> np.ndarray:
        """Compute probability density at points x = Vz.

        The points z are defined in normalized phase space (equal to
        regular phase space if V = I.
        """
        if z.ndim == 1:
            z = z[None, :]

        x = self.unnormalize(z)

        prob = np.ones(z.shape[0])
        for index, transform in enumerate(self.transforms):
            u = transform(x)
            for diag, lfunc in zip(self.diagnostics[index], self.lagrange_functions[index]):
                prob *= lfunc(diag.project(u))
                
        prob = prob * self.prior.prob(z)
        
        if squeeze:
            prob = np.squeeze(prob)
        return prob

    def sample(self, size: int, **kws) -> np.ndarray:
        """Sample `size` particles from the distribution.

        Key word arguments go to `self.sampler`.
        """
        def prob_func(z: np.ndarray) -> np.ndarray:
            return self.prob(z, squeeze=False)

        z = self.sampler(prob_func, size, **kws)
        return z

    def estimate_entropy(self, nsamp: float) -> float:
        """Estimate the relative entropy via Monte Carlo."""
        z = self.sample(nsamp)
        log_p = np.log(self.prob(z) + 1.00e-15)
        log_q = np.log(self.prior.prob(z) + 1.00e-15)
        entropy = -np.mean(log_p - log_q)
        return entropy

    def get_projection_points(self, index: int, diag_index: int) -> np.ndarray:
        """Return points on projection axis for specified diagnostic."""
        diagnostic = self.diagnostics[index][diag_index]
        return diagnostic.get_grid_points()

    def get_integration_points(self, index: int, diag_index: int, method: str = "grid") -> np.ndarray:
        """Return integration points for specific diagnnostic."""
        if self.integration_points is not None:
            return self.integration_points

        diagnostic = self.diagnostics[index][diag_index]

        projection_axis = diagnostic.axis
        if type(projection_axis) is int:
            projection_axis = (projection_axis,)

        integration_axis = tuple([axis for axis in range(self.ndim) if axis not in projection_axis])
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
                np.linspace(
                    integration_limits[i][0],
                    integration_limits[i][1],
                    integration_grid_shape[i],
                )
                for i in range(integration_ndim)
            ]
            if integration_ndim == 1:
                integration_points = integration_grid_coords[0]
            else:
                integration_points = get_grid_points(integration_grid_coords)
        else:
            raise NotImplementedError

        self.integration_points = integration_points
        return self.integration_points

    def simulate(self) -> list[list[Histogram]]:
        """Simulate all measurements."""
        diagnostic_copies = []
        for index in range(len(self.diagnostics)):
            diagnostic_copies.append([])
            for diag_index in range(len(self.diagnostics[index])):
                diagnostic_copy = self.simulate_single(index, diag_index)
                diagnostic_copies[-1].append(diagnostic_copy)
        return diagnostic_copies

    def simulate_single(self, index: int, diag_index: int) -> Histogram:
        """Simulate a single measurement.

        Parameters
        ----------
        index : int
            Transformation index.
        diag_index : int
            Diagnostic index for the given transformation.

        Returns
        -------
        Histogram
            Copy of updated histogram diagnostic.
        """
        transform = self.transforms[index]
        diagnostic = self.diagnostics[index][diag_index]
        diagnostic.values *= 0.0
        
        values_proj = diagnostic.values.copy()
        
        if self.mode in ["sample", "forward"]:
            values_proj = diagnostic(transform(self.unnormalize(self.sample(self.nsamp))))

        elif self.mode in ["integrate", "backward"]:
            # Get projection grid axis.
            projection_axis = diagnostic.axis
            if type(projection_axis) is int:
                projection_axis = (projection_axis,)
            projection_ndim = len(projection_axis)

            # Get integration grid axis and limits.
            integration_axis = [axis for axis in range(self.ndim) if axis not in projection_axis]
            integration_axis = tuple(integration_axis)
            integration_ndim = len(integration_axis)
            integration_limits = self.integration_limits[index][diag_index]

            # Get points on integration and projection grids.
            projection_points = self.get_projection_points(index, diag_index)
            integration_points = self.get_integration_points(index, diag_index)

            # Initialize array of integration points (u).
            u = np.zeros((integration_points.shape[0], self.ndim))
            for k, axis in enumerate(integration_axis):
                if integration_ndim == 1:
                    u[:, axis] = integration_points
                else:
                    u[:, axis] = integration_points[:, k]

            # Initialize array of projected densities (values_proj).
            values_proj = np.zeros(projection_points.shape[0])
            for i, point in enumerate(wrap_tqdm(projection_points, self.verbose > 1)):
                # Set values of u along projection axis.
                for k, axis in enumerate(projection_axis):
                    if diagnostic.ndim == 1:
                        u[:, axis] = point
                    else:
                        u[:, axis] = point[k]

                # Compute the probability density at the integration points.
                # Here we assume a volume-preserving transformation with Jacobian
                # determinant equal to 1, such that p(x) = p(u).
                prob = self.prob(self.normalize(transform.inverse(u)))

                # Sum over all integration points.
                values_proj[i] = np.sum(prob)

            # Reshape the projected density array.
            if diagnostic.ndim > 1:
                values_proj = values_proj.reshape(diagnostic.shape)

        else:
            raise ValueError(f"Invalid mode {self.mode}")

        # Update the diagnostic values.
        diagnostic.values = values_proj
        diagnostic.normalize()

        # Return a copy of the diagnostic.
        return diagnostic.copy()

    def gauss_seidel_step(
        self, 
        learning_rate: float = 1.0, 
        thresh: float = 0.0, 
        thresh_type: str = "abs",
    ) -> None:
        """Perform Gauss-Seidel update.

        The update is defined as:

            h *= 1.0 + omega * ((g_meas / g_pred) - 1.0)

        where h = exp(lambda) is the lagrange function, 0 < omega <= 1 is a learning
        rate or damping parameter, g_meas is the measured projection, and g_pred
        is the simulated projection.
        """
        for index, transform in enumerate(self.transforms):
            if self.verbose:
                print(f"transform={index}")

            for diag_index in range(len(self.diagnostics[index])):
                if self.verbose:
                    print(f"diagnostic={diag_index}")

                # Get lagrange multpliers, measured and simulated projections
                hist_pred = self.simulate_single(index=index, diag_index=diag_index)
                hist_meas = self.projections[index][diag_index]
                lagrange_function = self.lagrange_functions[index][diag_index]

                # Unravel
                values_lagr = lagrange_function.values.copy()
                values_meas = hist_meas.values.copy()
                values_pred = hist_pred.values.copy()

                # Threshold simulated projections (probably better to add to diagnostic)
                if thresh_type == "frac":
                    thresh = thresh * np.max(values_pred)
                values_pred[values_pred < thresh] = 0.0

                # Update lagrange multipliers
                idx = np.logical_and(values_meas > 0.0, values_pred > 0.0)
                ratio = np.ones(values_lagr.shape)
                ratio[idx] = values_meas[idx] / values_pred[idx]
                values_lagr *= 1.0 + learning_rate * (ratio - 1.0)

                # Reset
                lagrange_function.values = values_lagr
                lagrange_function.set_values(lagrange_function.values)
                self.lagrange_functions[index][diag_index] = lagrange_function

        self.epoch += 1

    def parameters(self) -> np.ndarray:
        """Return lagrange multplier values."""
        parameters = [lfunc.values.ravel() for lfunc in unravel(self.lagrange_functions)]
        parameters = np.hstack(parameters)
        return parameters

    def save(self, path: str) -> None:
        """Save model to pickled file."""
        state = {
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "projections": self.projections,
            "ndim": self.ndim,
            "prior": self.prior,
            "sampler": self.sampler,
            "unnorm_matrix": self.unnorm_matrix,
            "epoch": self.epoch,
            "lagrange_functions": self.lagrange_functions,
        }

        # [Q] Can we just do `pickle.dump(self, file)`?

        file = open(path, "wb")
        pickle.dump(state, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path: str) -> None:
        """Load model from pickled file."""
        file = open(path, "rb")

        state = pickle.load(file)

        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.projections = state["projections"]

        self.ndim = state["ndim"]
        self.prior = state["prior"]
        self.sampler = state["sampler"]
        self.unnorm_matrix = state["unnorm_matrix"]
        self.unnorm_transform = self.set_unnorm_transform(self.unnorm_matrix)

        self.epoch = state["epoch"]
        self.lagrange_functions = state["lagrange_functions"]

        file.close()
