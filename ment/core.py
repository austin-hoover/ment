import itertools
import pickle
from typing import Any
from typing import Callable

import numpy as np
import torch

from .diag import Histogram
from .diag import Histogram1D
from .interp import RegularGridInterpolator
from .interp import RegularGridInterpolationStencil
from .samp import GridSampler
from .prior import InfiniteUniformPrior
from .utils import get_grid_points
from .utils import wrap_tqdm
from .utils import unravel


class LagrangeFunction:
    """Represents exponential of Lagrange multiplier function on regular grid."""

    def __init__(self, projection: Histogram) -> None:
        self.projection = projection
        self.values = torch.zeros_like(self.projection.values)
        self.coords = None
        if type(projection) is Histogram1D:
            self.coords = [projection.coords]
        else:
            self.coords = projection.coords
        self.interp = RegularGridInterpolator(self.coords, self.values)

    def set_values(self, values: torch.Tensor) -> None:
        self.values = values
        self.interp = RegularGridInterpolator(self.coords, self.values)
        return self.values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.projection.project(x)
        return self.interp(x_proj)


class GridCache:
    """Caches MENT probability factors for use with GridSampler."""

    def __init__(
        self,
        sampler: Any,
        unnormalize: Callable,
        transforms: list[Callable],
        lagrange_functions: list[list[Callable]],
        prior: Any,
    ) -> None:
        self.sampler = sampler
        self.unnormalize = unnormalize
        self.transforms = transforms
        self.lagrange_functions = lagrange_functions
        self.prior = prior
        self.state = None

    def clear(self) -> None:
        self.state = None

    def build(self) -> dict:
        points = self.sampler.get_grid_points()
        if self.sampler.device is not None:
            points = points.to(self.sampler.device)

        x = self.unnormalize(points)
        interp_stencils = []
        interp_values = []
        for index, transform in enumerate(self.transforms):
            x_out = transform(x)
            interp_stencils.append([])
            interp_values.append([])
            for lagrange_function in self.lagrange_functions[index]:
                projected = lagrange_function.projection.project(x_out)
                stencil = RegularGridInterpolationStencil(
                    lagrange_function.coords,
                    projected,
                )
                interp_stencils[-1].append(stencil)
                interp_values[-1].append(stencil(lagrange_function.values))

        self.state = {
            "points": points,
            "interp_stencils": interp_stencils,
            "interp_values": interp_values,
            "prior_values": self.prior.prob(points),
            "prob_values": None,
        }
        self.refresh_prob()
        return self.state

    def ensure(self) -> dict:
        if self.state is None:
            return self.build()
        return self.state

    def refresh_prob(self) -> None:
        if self.state is None:
            return

        prob = self.state["prior_values"].clone()
        for values_by_transform in self.state["interp_values"]:
            for values in values_by_transform:
                prob *= values
        prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        self.state["prob_values"] = torch.clamp(prob, min=0.0)

    def refresh_lagrange(self, index: int, diag_index: int) -> None:
        if self.state is None:
            return

        lagrange_function = self.lagrange_functions[index][diag_index]
        stencil = self.state["interp_stencils"][index][diag_index]
        self.state["interp_values"][index][diag_index] = stencil(
            lagrange_function.values
        )
        self.refresh_prob()

    def sample(self, size: int) -> torch.Tensor:
        cache = self.ensure()
        values = cache["prob_values"]
        return self.sampler.sample_values(values, size)


class MENT:
    """Maximum Entropy Tomography (MENT) model."""

    def __init__(
        self,
        ndim: int,
        transforms: list[Callable],
        projections: list[list[Histogram]],
        prior: Any,
        sampler: Callable,
        unnorm_matrix: torch.Tensor = None,
        nsamp: int = 1_000_000,
        integration_limits: list[tuple[float, float]] = None,
        integration_size: int = None,
        integration_loop: bool = True,
        diag_kws: dict = None,
        cache_grid: bool = None,
        verbose: int = 1,
        mode: str = "sample",
    ) -> None:
        """Constructor.

        ndim:
            Number of phase space dimensions.
        transforms:
            Functions that transform the phase space coordinates. Call signature is
             `transform(x: torch.Tensor) -> torch.Tensor`, where `x` is a batch
             of shape (nsamp, ndim).
        projections:
            Measured projections.
        unnorm_matrix:
            Matrix that unnormalizes the phase space coordinates. Defaults to the
            identity matrix.
        prior:
            Prior distribution over the **normalized** phase space coordinates. Must
            implement `prior.prob(z: torch.Tensor) -> torch.Tensor`.
        sampler:
            Calling `sampler(p, n)` generates `n` samples from the PDF `p`.
        nsamp:
            Number of samples to use when computing projections. Only relevant if
            `self.mode=="sample".
        integration_limits:
            List of (min, max) coordinates of integration grid.
        integration_size:
            Number of integration points.
        integration_loop:
            If True, compute projection by looping over all points on the M-dimensional
            projection axis;  at each point, compute the (N - M)-dimensional integral.
            If False, compute projection by evaluating all points at once on an
            N-dimensional grid, then summing over the (N - M) integration axes.
        diag_kws:
            Key word arguments passed to `Histogram` constructor. Options include
            `blur`, `thresh`, and `thresh_type`.
        cache_grid:
            If True, cache probability values on a ``GridSampler`` grid and sample
            from those cached values. If None, this is enabled automatically for
            compatible grid samplers in sample/forward mode.
        verbose:
            Whether to print updates during calculations.
        mode:
            Whether to use numerical integration or particle sampling to compute
            projections. {"sample" or "forward", "integration" or "backward"}
        """
        self.ndim = ndim
        self.verbose = int(verbose)
        self.mode = mode

        # Set transforms and projection data
        self.transforms = transforms
        self.projections = self.set_projections(projections)

        # Setup histogram diagnostics.
        ## TODO: separate diagnostic class from Projection class? The Projection object
        ## just needs to store the bin coordinates and values. The Histogram object
        ## needs to bin the particles on the grid.
        if diag_kws is None:
            diag_kws = {}

        self.diagnostics = []
        for index in range(len(self.projections)):
            self.diagnostics.append([])
            for diag in self.projections[index]:
                diag_new = diag.copy()
                for key, val in diag_kws.items():
                    setattr(diag_new, key, val)
                self.diagnostics[-1].append(diag_new)

        # Prior distribution
        self.prior = prior
        if self.prior is None:
            self.prior = InfiniteUniformPrior(ndim=ndim)

        # Normalization matrix
        self.unnorm_matrix = unnorm_matrix
        self.set_unnorm_matrix(unnorm_matrix)

        # Initialize model parameters
        self.lagrange_functions = self.init_lagrange_functions()

        # Sampling
        self.sampler = sampler
        self.nsamp = int(nsamp)
        self.cache_grid = cache_grid
        if self.cache_grid is None:
            self.cache_grid = isinstance(self.sampler, GridSampler)
        self._grid_cache = None

        # Integration
        self.integration_limits = integration_limits
        self.integration_size = integration_size
        self.integration_points = None
        self.integration_loop = integration_loop

        self.iteration = 0

    def set_unnorm_matrix(self, unnorm_matrix: torch.Tensor) -> None:
        """Set normalization matrix.

        The inverse of the normalization matrix transforms the normalized
        coordiantes z to phase space coordinates x via the linear mapping
        x = Vz.

        The densities are related as p(x) = p(z) / det(V).
        """
        self.unnorm_matrix = unnorm_matrix
        if self.unnorm_matrix is None:
            self.unnorm_matrix = torch.eye(self.ndim)
        self.unnorm_matrix = self.unnorm_matrix.float()
        self.unnorm_matrix_det = torch.linalg.det(self.unnorm_matrix)
        self.norm_matrix = torch.linalg.inv(self.unnorm_matrix)
        self.norm_matrix_det = torch.linalg.det(self.norm_matrix)
        if hasattr(self, "_grid_cache") and self._grid_cache is not None:
            self._grid_cache.clear()

    def set_projections(
        self, projections: list[list[Histogram]]
    ) -> list[list[Histogram]]:
        """Set list of measured projections (histograms)."""
        self.projections = projections
        if self.projections is None:
            self.projections = [[]]
        return self.projections

    def init_lagrange_functions(self) -> list[list[LagrangeFunction]]:
        """Initialize lagrange multipler functions.

        The function h(u_proj) = 1 if the measured projection g(u_proj) > 0,
        otherwise h(u_proj) = 0.

        Key word arguments passed to `LagrangeFunction` constructor.
        """
        self.lagrange_functions = []
        for index in range(len(self.projections)):
            self.lagrange_functions.append([])
            for projection in self.projections[index]:
                values = torch.zeros(projection.shape)
                values[projection.values > 0.0] = 1.0
                lagrange_function = LagrangeFunction(projection)
                lagrange_function.set_values(values)
                self.lagrange_functions[-1].append(lagrange_function)
        return self.lagrange_functions

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Unnormalize coordinates z: x = Vz."""
        return torch.matmul(z, self.unnorm_matrix.T)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates x: z = V^-1 z."""
        return torch.matmul(x, self.norm_matrix.T)

    def prob(self, z: torch.Tensor, squeeze: bool = True) -> torch.Tensor:
        """Compute probability density at normalized coordinate z = V^-1 x.

        The points z are defined in normalized phase space (equal to regular
        phase space if V = I. This function returns the density up to a
        constant.
        """
        if z.ndim == 1:
            z = z[None, :]

        x = self.unnormalize(z)

        prob = torch.ones(z.shape[0])
        for index, transform in enumerate(self.transforms):
            x_out = transform(x)
            for lagrange_function in self.lagrange_functions[index]:
                prob = prob * lagrange_function(x_out)
        prob = prob * self.prior.prob(z)

        if squeeze:
            prob = torch.squeeze(prob)

        return prob

    def _grid_cache_enabled(self) -> bool:
        return (
            self.cache_grid
            and self.mode in ["sample", "forward"]
            and isinstance(self.sampler, GridSampler)
        )

    def _ensure_grid_cache(self) -> GridCache:
        if self._grid_cache is None:
            self._grid_cache = GridCache(
                sampler=self.sampler,
                unnormalize=self.unnormalize,
                transforms=self.transforms,
                lagrange_functions=self.lagrange_functions,
                prior=self.prior,
            )
        return self._grid_cache

    def _refresh_grid_cache_lagrange(self, index: int, diag_index: int) -> None:
        if self._grid_cache is None:
            return
        self._grid_cache.refresh_lagrange(index, diag_index)

    def _sample_grid_cache(self, size: int) -> torch.Tensor:
        return self._ensure_grid_cache().sample(size)

    @property
    def grid_cache(self) -> dict | None:
        if self._grid_cache is None:
            return None
        return self._grid_cache.state

    @grid_cache.setter
    def grid_cache(self, value: dict | None) -> None:
        if value is None:
            if self._grid_cache is not None:
                self._grid_cache.clear()
            return
        self._ensure_grid_cache().state = value

    def sample(self, size: int, **kws) -> torch.Tensor:
        """Sample `size` particles from the distribution in normalized space.

        To get the phase space coordinates, call `unnormalize(sample(size))`.

        Key word arguments go to `self.sampler`.
        """
        if self._grid_cache_enabled() and not kws:
            return self._sample_grid_cache(size)

        def prob_func(z: torch.Tensor) -> torch.Tensor:
            return self.prob(z, squeeze=False)

        z = self.sampler(prob_func, size, **kws)
        return z

    def _get_projection_points(self, index: int, diag_index: int) -> torch.Tensor:
        """Return points on projection axis for specified diagnostic."""
        diagnostic = self.diagnostics[index][diag_index]
        return diagnostic.get_grid_points()

    def _get_integration_points(
        self, index: int, diag_index: int, method: str = "grid"
    ) -> torch.Tensor:
        """Return integration points for specific diagnnostic."""
        if self.integration_points is not None:
            return self.integration_points

        diagnostic = self.diagnostics[index][diag_index]

        projection_axis = diagnostic.axis
        if type(projection_axis) is int:
            projection_axis = (projection_axis,)

        integration_axis = tuple(
            [axis for axis in range(self.ndim) if axis not in projection_axis]
        )
        integration_ndim = len(integration_axis)
        integration_limits = self.integration_limits[index][diag_index]
        integration_size = self.integration_size
        integration_points = None

        if (integration_ndim == 1) and (np.ndim(integration_limits) == 1):
            integration_limits = [integration_limits]

        if method == "grid":
            integration_grid_resolution = int(
                integration_size ** (1.0 / integration_ndim)
            )
            integration_grid_shape = tuple(
                integration_ndim * [integration_grid_resolution]
            )
            integration_grid_coords = [
                torch.linspace(
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

        Args:
            index: Transform index.
            diag_index: Diagnostic index for this transform.

        Returns:
            Copy of updated histogram diagnostic.
        """
        transform = self.transforms[index]
        diagnostic = self.diagnostics[index][diag_index]
        diagnostic.values *= 0.0

        if self.mode in ["sample", "forward"]:
            values_proj = diagnostic(
                transform(self.unnormalize(self.sample(self.nsamp)))
            )

        elif self.mode in ["integrate", "reverse"]:
            # Get projection grid axis.
            projection_axis = diagnostic.axis
            if type(projection_axis) is int:
                projection_axis = (projection_axis,)

            # Get integration grid axis and limits.
            integration_axis = [
                axis for axis in range(self.ndim) if axis not in projection_axis
            ]
            integration_axis = tuple(integration_axis)
            integration_ndim = len(integration_axis)

            if self.integration_loop:
                # Get points on integration and projection grids.
                projection_points = self._get_projection_points(index, diag_index)
                integration_points = self._get_integration_points(index, diag_index)

                # Initialize array of integration points (x_out).
                x_out = torch.zeros((integration_points.shape[0], self.ndim))
                for k, axis in enumerate(integration_axis):
                    if integration_ndim == 1:
                        x_out[:, axis] = integration_points
                    else:
                        x_out[:, axis] = integration_points[:, k]

                # Initialize array of projected densities (values_proj).
                values_proj = torch.zeros(projection_points.shape[0])
                for i, point in enumerate(
                    wrap_tqdm(projection_points, self.verbose > 1)
                ):
                    # Set values of x_out along projection axis.
                    for k, axis in enumerate(projection_axis):
                        if diagnostic.ndim == 1:
                            x_out[:, axis] = point
                        else:
                            x_out[:, axis] = point[k]

                    # Compute the probability density at the integration points.
                    # Here we assume a volume-preserving transformation with Jacobian
                    # determinant equal to 1, such that p(x) = p(u).
                    prob = self.prob(self.normalize(transform.inverse(x_out)))

                    # Sum over all integration points.
                    values_proj[i] = torch.sum(prob)

            else:
                # Evaluate all points at once on N-dimensional grid, then sum over
                # integration axes.
                grid_coords = [None] * self.ndim

                # Get coordinates along each axis of projection grid.
                projection_grid_coords = diagnostic.coords
                if np.ndim(projection_grid_coords[0]) == 0:
                    projection_grid_coords = [projection_grid_coords]

                # Get coordinates along each axis of integration grid.
                integration_axis = [
                    axis for axis in range(self.ndim) if axis not in projection_axis
                ]
                integration_axis = tuple(integration_axis)
                integration_ndim = len(integration_axis)
                integration_limits = self.integration_limits[index][diag_index]
                if (integration_ndim == 1) and (np.ndim(integration_limits) == 1):
                    integration_limits = [integration_limits]

                integration_grid_resolution = int(
                    self.integration_size ** (1.0 / integration_ndim)
                )
                integration_grid_shape = tuple(
                    integration_ndim * [integration_grid_resolution]
                )
                integration_grid_coords = [
                    torch.linspace(
                        integration_limits[i][0],
                        integration_limits[i][1],
                        integration_grid_shape[i],
                    )
                    for i in range(integration_ndim)
                ]

                # Create N-dimensional meshgrid
                for i, _coords in zip(projection_axis, projection_grid_coords):
                    grid_coords[i] = _coords

                for i, _coords in zip(integration_axis, integration_grid_coords):
                    grid_coords[i] = _coords

                grid_shape = tuple([len(c) for c in grid_coords])
                grid_points = get_grid_points(grid_coords)
                grid_values = self.prob(self.normalize(transform.inverse(grid_points)))
                grid_values = grid_values.reshape(grid_shape)
                values_proj = torch.sum(grid_values, axis=integration_axis)

            # Reshape the projected density array.
            if diagnostic.ndim > 1:
                values_proj = values_proj.reshape(diagnostic.shape)

        else:
            raise ValueError(f"Invalid mode {self.mode}")

        # Update the diagnostic values and return a copy.
        diagnostic.values = values_proj
        diagnostic.process()
        return diagnostic.copy()

    def gauss_seidel_step(
        self, lr: float = 1.0, thresh: float = 0.0, thresh_type: str = "frac"
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

                # Get Lagrange multipliers, measured and simulated projections
                hist_pred = self.simulate_single(index=index, diag_index=diag_index)
                hist_meas = self.projections[index][diag_index]
                lagrange_function = self.lagrange_functions[index][diag_index]

                # Unravel values array
                values_lagr = torch.clone(lagrange_function.values)
                values_meas = torch.clone(hist_meas.values)
                values_pred = torch.clone(hist_pred.values)

                min_value = 0.0
                if thresh_type == "frac":
                    min_value = thresh * torch.max(values_pred)
                else:
                    min_value = thresh

                # Update Lagrange multipliers
                idx = torch.logical_and(
                    values_meas > min_value, values_pred > min_value
                )
                ratio = torch.ones(values_lagr.shape)
                ratio[idx] = values_meas[idx] / values_pred[idx]
                values_lagr *= 1.0 + lr * (ratio - 1.0)

                # Reset
                lagrange_function.values = values_lagr
                lagrange_function.set_values(lagrange_function.values)
                self.lagrange_functions[index][diag_index] = lagrange_function
                self._refresh_grid_cache_lagrange(index, diag_index)

        self.iteration += 1

    def parameters(self) -> torch.Tensor:
        """Return vector of Lagrange multipliers."""
        parameters = [
            lfunc.values.ravel() for lfunc in unravel(self.lagrange_functions)
        ]
        parameters = torch.hstack(parameters)
        return parameters

    def save(self, path: str) -> None:
        """Save model to file."""
        state = {
            "lagrange_functions": self.lagrange_functions,
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "projections": self.projections,
            "ndim": self.ndim,
            "prior": self.prior,
            "sampler": self.sampler,
            "unnorm_matrix": self.unnorm_matrix,
            "iteration": self.iteration,
            "cache_grid": self.cache_grid,
        }

        # Can we just do `pickle.dump(self, file)`?
        file = open(path, "wb")
        pickle.dump(state, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path: str) -> None:
        """Load model from file."""
        file = open(path, "rb")

        state = pickle.load(file)

        self.lagrange_functions = state["lagrange_functions"]
        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.projections = state["projections"]

        self.ndim = state["ndim"]
        self.prior = state["prior"]
        self.sampler = state["sampler"]
        self.set_unnorm_matrix(state["unnorm_matrix"])
        self.cache_grid = state.get("cache_grid", isinstance(self.sampler, GridSampler))
        self._grid_cache = None

        file.close()
