"""Covariance matrix fitting."""

from collections.abc import Callable, Sequence

import numpy as np
import scipy.optimize
import torch
from scipy.optimize import Bounds, OptimizeResult

from ..diag import Histogram1D, HistogramND
from ..sim import (
    ComposedTransform,
    IdentityTransform,
    LinearTransform,
    ProjectionTransform1D,
)

Diagnostic = Histogram1D | HistogramND


class CovFitterBase:
    """Base class for fitting a Gaussian covariance matrix to measured moments.

    Linear transforms and the built-in histogram diagnostics are evaluated
    analytically by default. Other transforms fall back to Monte Carlo samples.
    """

    def __init__(
        self,
        ndim: int,
        transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        projections: Sequence[Sequence[Diagnostic]],
        nsamp: int,
        unnorm_matrix: torch.Tensor | None = None,
        verbose: int | bool = 2,
        loss_scale: float = 1.0,
        resample: bool = True,
        seed: int | None = None,
        loss_mode: str = "auto",
    ) -> None:
        if ndim < 1:
            raise ValueError("ndim must be positive.")
        if nsamp < 2:
            raise ValueError("nsamp must be at least 2.")
        if len(transforms) != len(projections):
            raise ValueError("transforms and projections must have the same length.")
        if not projections or not any(projections):
            raise ValueError("At least one projection is required.")

        self.ndim = int(ndim)
        self.nsamp = int(nsamp)
        self.verbose = int(verbose)
        self.resample = bool(resample)

        if unnorm_matrix is None:
            unnorm_matrix = torch.eye(self.ndim)
        self.unnorm_matrix = torch.as_tensor(unnorm_matrix, dtype=torch.float32)
        if self.unnorm_matrix.shape != (self.ndim, self.ndim):
            raise ValueError(
                "unnorm_matrix must have shape "
                f"({self.ndim}, {self.ndim}), got {self.unnorm_matrix.shape}."
            )

        self.params = np.empty(0, dtype=float)
        self.lb = np.empty(0, dtype=float)
        self.ub = np.empty(0, dtype=float)

        self.loss_scale = float(loss_scale)
        self.transforms = list(transforms)
        self.projections = [list(group) for group in projections]
        self.diagnostics = [
            [projection.copy() for projection in group] for group in projections
        ]

        # Compute histogram moments for fitting.
        self._target_moments = [
            [self._histogram_moment(projection) for projection in group]
            for group in self.projections
        ]
        self._transform_matrices = self._get_transform_matrices()
        self.loss_mode = self._resolve_loss_mode(loss_mode)

        self._generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=self.unnorm_matrix.device)
            generator.manual_seed(seed)
            self._generator = generator
        self._base_samples: torch.Tensor | None = None

        self.iteration = 0
        self.nevals = 0
        self.loss: float | None = None
        self.best_loss = np.inf
        self.best_params: np.ndarray | None = None

    @staticmethod
    def _histogram_moment(histogram: Diagnostic) -> torch.Tensor:
        if isinstance(histogram, Histogram1D):
            return histogram.var().detach().clone()
        return histogram.cov().detach().clone()

    def _resolve_loss_mode(self, loss_mode: str) -> str:
        if loss_mode not in {"auto", "analytic", "sample"}:
            raise ValueError("loss_mode must be 'auto', 'analytic', or 'sample'.")

        analytic_supported = self._transform_matrices is not None and all(
            type(diagnostic) in {Histogram1D, HistogramND}
            for group in self.diagnostics
            for diagnostic in group
        )
        if loss_mode == "analytic" and not analytic_supported:
            raise ValueError(
                "Analytic loss requires built-in linear transforms and histogram "
                "diagnostics. Use loss_mode='sample' for nonlinear transforms."
            )
        if loss_mode == "auto":
            return "analytic" if analytic_supported else "sample"
        return loss_mode

    def _get_transform_matrices(self) -> list[torch.Tensor] | None:
        matrices = []
        for transform in self.transforms:
            matrix = self._get_transform_matrix(transform, self.ndim)
            if matrix is None:
                return None
            matrices.append(matrix)
        return matrices

    @classmethod
    def _get_transform_matrix(
        cls, transform: Callable, input_dim: int
    ) -> torch.Tensor | None:
        if isinstance(transform, IdentityTransform):
            return torch.eye(input_dim)
        if isinstance(transform, LinearTransform):
            matrix = torch.as_tensor(transform.matrix)
            if matrix.ndim != 2 or matrix.shape[1] != input_dim:
                return None
            return matrix
        if isinstance(transform, ProjectionTransform1D):
            direction = torch.as_tensor(transform.direction)
            if direction.shape != (input_dim,):
                return None
            return direction[None, :]
        if isinstance(transform, ComposedTransform):
            matrix = torch.eye(input_dim)
            output_dim = input_dim
            for child in transform.transforms:
                child_matrix = cls._get_transform_matrix(child, output_dim)
                if child_matrix is None:
                    return None
                matrix = child_matrix @ matrix.to(child_matrix)
                output_dim = child_matrix.shape[0]
            return matrix
        return None

    def set_params(self, params: np.ndarray) -> None:
        params = np.asarray(params, dtype=float)
        if self.params.size and params.shape != self.params.shape:
            raise ValueError(
                f"Expected {self.params.size} parameters, got {params.size}."
            )
        self.params = np.clip(params, self.lb, self.ub)

    def build_cov(self) -> torch.Tensor:
        normalized_cov = self._build_cov()
        return self.unnorm_matrix @ normalized_cov @ self.unnorm_matrix.T

    def _build_cov(self) -> torch.Tensor:
        raise NotImplementedError

    def _build_factor(self) -> torch.Tensor:
        return torch.linalg.cholesky(self._build_cov())

    def set_cov(self, cov_matrix: torch.Tensor) -> None:
        """Set parameters from a covariance matrix in physical coordinates."""
        raise NotImplementedError

    def _normalized_covariance(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        cov_matrix = torch.as_tensor(
            cov_matrix,
            dtype=self.unnorm_matrix.dtype,
            device=self.unnorm_matrix.device,
        )
        if cov_matrix.shape != (self.ndim, self.ndim):
            raise ValueError(
                "cov_matrix must have shape "
                f"({self.ndim}, {self.ndim}), got {cov_matrix.shape}."
            )
        normalized_cov = torch.linalg.solve(self.unnorm_matrix, cov_matrix)
        normalized_cov = torch.linalg.solve(self.unnorm_matrix, normalized_cov.T).T
        return 0.5 * (normalized_cov + normalized_cov.T)

    def _draw_standard_normal(self, size: int, reuse: bool) -> torch.Tensor:
        shape = (size, self.ndim)
        if reuse and self._base_samples is not None:
            if self._base_samples.shape == shape:
                return self._base_samples

        samples = torch.randn(
            shape,
            dtype=self.unnorm_matrix.dtype,
            device=self.unnorm_matrix.device,
            generator=self._generator,
        )
        if reuse:
            self._base_samples = samples
        return samples

    def sample(self, size: int | None = None) -> torch.Tensor:
        """Sample the Gaussian represented by the current parameters."""
        reuse = size is None and not self.resample
        if size is None:
            size = self.nsamp
        if size < 1:
            raise ValueError("size must be positive.")

        # Building samples from the fitted factor avoids decomposing the full
        # covariance on every objective evaluation.
        factor = self.unnorm_matrix @ self._build_factor()
        standard_normal = self._draw_standard_normal(size, reuse=reuse)
        return standard_normal @ factor.T

    @staticmethod
    def _sample_covariance(samples: torch.Tensor) -> torch.Tensor:
        centered = samples - torch.mean(samples, dim=0, keepdim=True)
        return centered.T @ centered / (samples.shape[0] - 1)

    @staticmethod
    def _diagnostic_covariance(
        covariance: torch.Tensor, diagnostic: Diagnostic
    ) -> torch.Tensor:
        if isinstance(diagnostic, Histogram1D):
            if diagnostic.direction is None:
                return covariance[diagnostic.axis, diagnostic.axis]
            direction = diagnostic.direction.to(covariance)
            return direction @ covariance @ direction

        axes = torch.as_tensor(
            diagnostic.axis, dtype=torch.long, device=covariance.device
        )
        return covariance.index_select(0, axes).index_select(1, axes)

    @staticmethod
    def _moment_residual(
        predicted: torch.Tensor, measured: torch.Tensor
    ) -> torch.Tensor:
        measured = measured.to(predicted)
        return predicted - measured

    def _analytic_residuals(self) -> list[torch.Tensor]:
        covariance = self.build_cov()
        residuals = []
        assert self._transform_matrices is not None
        for matrix, diagnostics, targets in zip(
            self._transform_matrices, self.diagnostics, self._target_moments
        ):
            matrix = matrix.to(covariance)
            output_covariance = matrix @ covariance @ matrix.T
            for diagnostic, target in zip(diagnostics, targets):
                predicted = self._diagnostic_covariance(output_covariance, diagnostic)
                residuals.append(self._moment_residual(predicted, target))
        return residuals

    def _sample_residuals(self) -> list[torch.Tensor]:
        samples = self.sample()
        residuals = []
        for transform, diagnostics, targets in zip(
            self.transforms, self.diagnostics, self._target_moments
        ):
            transformed = transform(samples)
            for diagnostic, target in zip(diagnostics, targets):
                projected = diagnostic.project(transformed)
                if diagnostic.ndim == 1:
                    predicted = torch.var(projected.reshape(-1))
                else:
                    predicted = self._sample_covariance(projected)
                residuals.append(self._moment_residual(predicted, target))
        return residuals

    def _evaluate_residuals(self) -> list[torch.Tensor]:
        return (
            self._analytic_residuals()
            if self.loss_mode == "analytic"
            else self._sample_residuals()
        )

    def _record_evaluation(self, loss: float) -> None:
        self.loss = loss
        self.nevals += 1
        if self.verbose > 2:
            print(f"loss={loss:0.4e} evals={self.nevals}")

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = np.array(self.params, copy=True)

    def loss_function(self, params: np.ndarray) -> float:
        """Return the mean absolute moment error for ``params``."""
        self.set_params(params)
        with torch.inference_mode():
            residuals = self._evaluate_residuals()
            loss = float(
                torch.stack([torch.mean(torch.abs(value)) for value in residuals])
                .mean()
                .mul(self.loss_scale)
            )
        self._record_evaluation(loss)
        return loss

    def residual_function(self, params: np.ndarray) -> np.ndarray:
        """Return dimension-balanced residuals for least-squares fitting."""
        self.set_params(params)
        with torch.inference_mode():
            residuals = self._evaluate_residuals()
            loss = float(
                torch.stack([torch.mean(torch.abs(value)) for value in residuals])
                .mean()
                .mul(self.loss_scale)
            )
            scale = self.loss_scale / np.sqrt(len(residuals))
            vector = torch.cat(
                [
                    residual.reshape(-1) / np.sqrt(residual.numel())
                    for residual in residuals
                ]
            ).mul(scale)
        self._record_evaluation(loss)
        return vector.detach().cpu().numpy()

    def _reset_fit_state(self) -> None:
        self.iteration = 0
        self.nevals = 0
        self.loss = None
        self.best_loss = np.inf
        self.best_params = None

    def _report_iteration(self, params: np.ndarray) -> None:
        self.set_params(params)
        self.iteration += 1
        loss = self.loss if self.loss is not None else np.nan
        if self.verbose > 0:
            print(f"iter={self.iteration:04d} loss={loss:0.4e} " f"evals={self.nevals}")
        if self.verbose > 1 and self.ndim < 6:
            print("cov_matrix:")
            print(self.build_cov())

    def fit(
        self, method: str = "differential-evolution", iters: int = 500, **opt_kws
    ) -> tuple[torch.Tensor, OptimizeResult]:
        """Fit the covariance parameters with a SciPy optimizer."""
        method = method.lower()
        self._reset_fit_state()
        bounds = Bounds(self.lb, self.ub)

        if method in {"nelder-mead", "powell", "l-bfgs-b"}:
            options = dict(opt_kws.pop("options", {}))
            options.setdefault("disp", bool(self.verbose))
            options.setdefault("maxiter", iters)
            if method == "l-bfgs-b":
                options.setdefault("eps", 1.00e-4)
            result = scipy.optimize.minimize(
                self.loss_function,
                self.params,
                method=method,
                bounds=bounds,
                callback=lambda params: self._report_iteration(params),
                options=options,
                **opt_kws,
            )
        elif method == "least-squares":
            opt_kws.setdefault("verbose", 2 if self.verbose else 0)
            opt_kws.setdefault("xtol", 1.00e-15)
            opt_kws.setdefault("ftol", 1.00e-15)
            opt_kws.setdefault("gtol", 1.00e-15)
            opt_kws.setdefault("max_nfev", iters)
            opt_kws.setdefault("diff_step", 1.00e-4)
            opt_kws.setdefault("bounds", (self.lb, self.ub))
            result = scipy.optimize.least_squares(
                self.residual_function, self.params, **opt_kws
            )
        elif method == "differential-evolution":
            opt_kws.setdefault("popsize", 5)
            opt_kws.setdefault("disp", bool(self.verbose))
            opt_kws.setdefault("maxiter", iters)
            result = scipy.optimize.differential_evolution(
                self.loss_function,
                bounds,
                callback=lambda intermediate_result: self._report_iteration(
                    intermediate_result.x
                ),
                x0=self.params,
                **opt_kws,
            )
        elif method == "dual-annealing":
            opt_kws.setdefault("maxiter", iters)
            opt_kws.setdefault("x0", self.params)
            result = scipy.optimize.dual_annealing(
                self.loss_function,
                bounds,
                callback=lambda params, _loss, _context: self._report_iteration(params),
                **opt_kws,
            )
        elif method == "shgo":
            opt_kws.setdefault("iters", iters)
            result = scipy.optimize.shgo(
                self.loss_function,
                bounds,
                callback=lambda params: self._report_iteration(params),
                **opt_kws,
            )
        elif method == "direct":
            opt_kws.setdefault("maxiter", iters)
            opt_kws.setdefault("vol_tol", 1.00e-100)
            opt_kws.setdefault("len_tol", 1.00e-18)
            result = scipy.optimize.direct(
                self.loss_function,
                bounds,
                callback=lambda params: self._report_iteration(params),
                **opt_kws,
            )
        else:
            raise ValueError(f"Unknown optimization method {method}.")

        self.set_params(result.x)
        return self.build_cov(), result


class CholeskyCovFitter(CovFitterBase):
    """Parameterize normalized covariance as ``S = L @ L.T``."""

    def __init__(self, bound: float = 1.00e15, resample: bool = True, **kwargs) -> None:
        super().__init__(resample=resample, **kwargs)
        self.nparam = self.ndim * (self.ndim + 1) // 2

        self.L = torch.eye(
            self.ndim,
            dtype=self.unnorm_matrix.dtype,
            device=self.unnorm_matrix.device,
        )
        self.idx_diag = (np.arange(self.ndim), np.arange(self.ndim))
        self.idx_offdiag = np.tril_indices(self.ndim, k=-1)
        self._idx_diag = torch.arange(self.ndim, device=self.L.device)
        self._idx_offdiag = tuple(
            torch.as_tensor(index, dtype=torch.long, device=self.L.device)
            for index in self.idx_offdiag
        )

        self.ub = np.full(self.nparam, bound, dtype=float)
        self.lb = -self.ub
        self.lb[: self.ndim] = 1.00e-15

        self.params = np.zeros(self.nparam, dtype=float)
        self.params[: self.ndim] = 1.0
        self.set_params(self.params)

    def _build_factor(self) -> torch.Tensor:
        params = torch.as_tensor(self.params, dtype=self.L.dtype, device=self.L.device)
        self.L.zero_()
        self.L[self._idx_diag, self._idx_diag] = params[: self.ndim]
        self.L[self._idx_offdiag] = params[self.ndim :]
        return self.L

    def _build_cov(self) -> torch.Tensor:
        factor = self._build_factor()
        return factor @ factor.T

    def set_cov(self, cov_matrix: torch.Tensor) -> None:
        factor = torch.linalg.cholesky(self._normalized_covariance(cov_matrix))
        params = np.empty(self.nparam, dtype=float)
        params[: self.ndim] = (
            factor[self._idx_diag, self._idx_diag].detach().cpu().numpy()
        )
        params[self.ndim :] = factor[self._idx_offdiag].detach().cpu().numpy()
        self.set_params(params)

    def set_bounds(self, bound: float) -> None:
        if bound <= 0:
            raise ValueError("bound must be positive.")
        self.ub = np.full(self.nparam, bound, dtype=float)
        self.lb = -self.ub
        self.lb[: self.ndim] = 1.00e-15
        self.set_params(self.params)


class LinearCovFitter(CovFitterBase):
    """Parameterize normalized covariance with an unconstrained square factor."""

    def __init__(self, bound: float = 1.00e15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nparam = self.ndim**2
        self.ub = np.full(self.nparam, +bound, dtype=float)
        self.lb = np.full(self.nparam, -bound, dtype=float)
        self.params = np.ravel(np.eye(self.ndim))
        self.set_params(self.params)

    def get_unnorm_matrix(self) -> torch.Tensor:
        return torch.as_tensor(
            self.params,
            dtype=self.unnorm_matrix.dtype,
            device=self.unnorm_matrix.device,
        ).reshape(self.ndim, self.ndim)

    def _build_factor(self) -> torch.Tensor:
        return self.get_unnorm_matrix()

    def _build_cov(self) -> torch.Tensor:
        factor = self._build_factor()
        return factor @ factor.T

    def set_cov(self, cov_matrix: torch.Tensor) -> None:
        factor = torch.linalg.cholesky(self._normalized_covariance(cov_matrix))
        self.set_params(factor.detach().cpu().numpy().ravel())
