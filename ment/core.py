from typing import Callable
from typing import Union
from typing import Any

import itertools
import numpy as np
import torch

from .diag import Histogram
from .prior import InfiniteUniformPrior
from .sim import IdentityTransform
from .sim import LinearTransform
from .utils import get_grid_points
from .utils import wrap_tqdm


class RegularGridInterpolator:
    """Interpolate points on a regular grid.

    https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py
    """

    def __init__(
        self, coords: list[torch.Tensor], values: torch.Tensor, fill_value: float = 0.0
    ) -> None:
        self.coords = coords
        self.values = values
        self.fill_value = torch.tensor(fill_value)

        assert isinstance(self.coords, tuple) or isinstance(self.coords, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.coords)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.coords):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, new_points: torch.Tensor) -> torch.Tensor:
        assert self.coords is not None
        assert self.values is not None

        assert len(new_points) == len(self.coords)
        K = new_points[0].shape[0]
        for x in new_points:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.coords, new_points):
            idx_right = torch.bucketize(x.contiguous(), p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.0
            dist_right[dist_right < 0] = 0.0
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.0

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.0
        for indexer in itertools.product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[tuple(as_s)] * torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        result = numerator / denominator

        # Handle bounds
        out_of_bounds = torch.zeros(
            new_points.shape[1], dtype=torch.bool, device=self.values.device
        )
        for x, c in zip(new_points, self.coords):
            out_of_bounds = out_of_bounds | (x < c[0]) | (x > c[-1])
        result[out_of_bounds] = self.fill_value

        return result


class LagrangeFunction:
    """Represents Lagrange multiplier function on regular grid.

    This function can be evaluated at any point by interpolation.
    """

    def __init__(
        self,
        ndim: int,
        axis: tuple[int] | int,
        coords: list[torch.Tensor] | torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Constructor.

        Args:
            ndim: Number of dimensions.
            axis: Defines grid axis. Example: (0, 1).
            coords: Grid coordinates along each dimension.
            values: Grid values.
        """
        self.ndim = ndim
        self.axis = axis

        if type(self.axis) is int:
            self.axis = (self.axis,)

        self.coords = coords
        if type(self.coords) is torch.Tensor:
            self.coords = [self.coords]

        self.values = self.set_values(values)

        self.limits = [(c[0], c[-1]) for c in self.coords]
        self.limits = torch.tensor(self.limits)

    def set_values(self, values: torch.Tensor) -> None:
        """Set grid values."""
        self.values = values
        self.interp = RegularGridInterpolator(self.coords, self.values)
        return self.values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate grid values at points x.

        x is a batch of points, shape (nsamp, ndim).
        """
        x_proj = x[:, self.axis]
        return self.interp(x_proj.T)


class MENT:
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
        store_integration_points: bool = True,
        verbose: int = 1,
        mode: str = "sample",
    ) -> None:
        """Constructor.

        ndim:
            Number of phase space dimensions.
        transforms:
            Functions that transform the phase space coordinates. Call signature is
             `transform(x: torch.Tensor) -> torch.Tensor`.
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
        store_integration_points:
            Whether to keep the integration points in memory.
        interpolation_kws:
            Key word arguments passed to `scipy.interpolate.RegularGridInterpolator` for
            interpolating the Lagrange multiplier functions.
        verbose:
            Whether to print updates during calculations.
        mode:
            Whether to use numerical integration or particle sampling to compute
            projections. {"sample" or "forward", "integration" or "backward"}
        """
        self.ndim = ndim
        self.verbose = int(verbose)
        self.mode = mode

        self.transforms = transforms
        self.projections = self.set_projections(projections)

        self.diagnostics = []
        for index in range(len(self.projections)):
            self.diagnostics.append([hist.copy() for hist in self.projections[index]])

        self.prior = prior
        if self.prior is None:
            self.prior = InfiniteUniformPrior(ndim=ndim)

        self.unnorm_matrix = unnorm_matrix
        self.unnorm_transform = self.set_unnorm_transform(unnorm_matrix)

        self.lagrange_functions = self.init_lagrange_functions()

        self.sampler = sampler
        self.nsamp = int(nsamp)

        self.integration_limits = integration_limits
        self.integration_size = integration_size
        self.integration_points = None
        self.integration_loop = integration_loop
        self.store_integration_points = store_integration_points

        self.iteration = 0

    def set_unnorm_transform(self, unnorm_matrix: torch.Tensor) -> None:
        """Set inverse of normalization matrix.

        The unnormalization matrix transforms normalized coordinates z to
        phase space coordinates x via the linear mapping: x = Vz.
        """
        self.unnorm_matrix = unnorm_matrix
        if self.unnorm_matrix is None:
            self.unnorm_transform = IdentityTransform()
            self.unnorm_matrix = torch.eye(self.ndim)
        else:
            self.unnorm_transform = LinearTransform(self.unnorm_matrix)

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
                lagrange_function = LagrangeFunction(
                    ndim=projection.ndim,
                    axis=projection.axis,
                    coords=projection.coords,
                    values=values,
                )
                self.lagrange_functions[-1].append(lagrange_function)
        return self.lagrange_functions

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Unnormalize coordinates z: x = Vz."""
        if self.unnorm_transform is None:
            self.unnorm_transform = IdentityTransform()
        return self.unnorm_transform(z)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates x: z = V^-1 z."""
        return self.unnorm_transform.inverse(x)

    def prob(self, z: torch.Tensor, squeeze: bool = True) -> torch.Tensor:
        """Compute probability density at points x = Vz.

        The points z are defined in normalized phase space (equal to
        regular phase space if V = I.
        """
        if z.ndim == 1:
            z = z[None, :]

        x = self.unnormalize(z)

        prob = torch.ones(z.shape[0])
        for index, transform in enumerate(self.transforms):
            u = transform(x)
            for lagrange_function in self.lagrange_functions[index]:
                prob = prob * lagrange_function(u)
        prob = prob * self.prior.prob(z)

        if squeeze:
            prob = torch.squeeze(prob)

        return prob

    def sample(self, size: int, **kws) -> torch.Tensor:
        """Sample `size` particles from the distribution.

        Key word arguments go to `self.sampler`.
        """

        def prob_func(z: torch.Tensor) -> torch.Tensor:
            return self.prob(z, squeeze=False)

        z = self.sampler(prob_func, size, **kws)
        return z

    def get_projection_points(self, index: int, diag_index: int) -> torch.Tensor:
        """Return points on projection axis for specified diagnostic."""
        diagnostic = self.diagnostics[index][diag_index]
        return diagnostic.get_grid_points()

    def get_integration_points(
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
                projection_points = self.get_projection_points(index, diag_index)
                integration_points = self.get_integration_points(index, diag_index)

                # Initialize array of integration points (u).
                u = torch.zeros((integration_points.shape[0], self.ndim))
                for k, axis in enumerate(integration_axis):
                    if integration_ndim == 1:
                        u[:, axis] = integration_points
                    else:
                        u[:, axis] = integration_points[:, k]

                # Initialize array of projected densities (values_proj).
                values_proj = torch.zeros(projection_points.shape[0])
                for i, point in enumerate(
                    wrap_tqdm(projection_points, self.verbose > 1)
                ):
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
        diagnostic.normalize()
        return diagnostic.copy()

    def gauss_seidel_step(
        self, learning_rate: float = 1.0, thresh: float = 0.0, thresh_type: str = "frac"
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
                values_lagr *= 1.0 + learning_rate * (ratio - 1.0)

                # Reset
                lagrange_function.values = values_lagr
                lagrange_function.set_values(lagrange_function.values)
                self.lagrange_functions[index][diag_index] = lagrange_function

        self.iteration += 1
