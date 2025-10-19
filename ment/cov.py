"""Covariance matrix analysis/fitting."""
import math
from typing import Callable

import torch
import scipy.optimize
from scipy.optimize import OptimizeResult
from scipy.optimize import Bounds

from .diag import Histogram


def normalize_eigvec(v: torch.Tensor) -> torch.Tensor:
    """Normalize eigenvectors according to Lebedev-Bogacz convention.

    conj(v)^T U v = 2i
    """
    ndim = len(v)
    v = torch.clone(torch.as_tensor(v))
    U = build_poisson_matrix(ndim=ndim, complex=True)

    def norm(v: torch.Tensor) -> torch.Tensor:
        return torch.linalg.multi_dot([torch.conj(v), U, v])

    if torch.imag(norm(v)) > 0:
        v = torch.conj(v)

    v *= torch.sqrt(2.0 / torch.abs(norm(v)))
    assert torch.isclose(torch.imag(norm(v)), -torch.tensor(2.0))
    assert torch.isclose(torch.real(norm(v)), +torch.tensor(0.0))
    return v


def build_poisson_matrix(ndim: int, complex: bool = False) -> torch.Tensor:
    """Return 4 x 4 Poisson matrix (assumes x-x' ordering)."""
    U = torch.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i, i + 1] = +1.0
        U[i + 1, i] = -1.0
    if complex:
        U = torch.complex(U, torch.zeros_like(U))
    return U


def build_norm_matrix_from_eigvecs(*eigvecs: list[torch.Tensor]) -> torch.Tensor:
    """Return normalization matrix V^-1 from eigenvectors."""
    ndim = eigvecs[0].shape[0]
    V = torch.zeros((ndim, ndim))
    for i, v in enumerate(eigvecs):
        V[:, i * 2 + 0] = +v.real
        V[:, i * 2 + 1] = -v.imag
    return torch.linalg.inv(V)


def build_scale_matrix(emittances: torch.Tensor) -> torch.Tensor:
    """Return 4 x 4 emittance scaling matrix."""
    diagonal = torch.clone(torch.as_tensor(emittances))
    diagonal = torch.sqrt(torch.repeat_interleave(diagonal, 2))
    return torch.diag(diagonal)


def build_norm_matrix_from_cov(
    cov_matrix: torch.Tensor, scale: bool = False
) -> torch.Tensor:
    """Return 4 x 4 symplectic normalization matrix from covariance matrix."""
    S = cov_matrix
    U = build_poisson_matrix(cov_matrix.shape[0])
    U = U.to(S.device)
    SU = torch.matmul(S, U)

    eigvals, eigvecs = torch.linalg.eig(SU)

    idx = eigvals.imag > 0.0
    eigvecs = eigvecs[:, idx]

    eigvecs = eigvecs.T
    for i, v in enumerate(eigvecs):
        eigvecs[i, :] = normalize_eigvec(v)

    V_inv = build_norm_matrix_from_eigvecs(*eigvecs)
    if scale:
        A = torch.linalg.multi_dot([V_inv, S, V_inv.T])
        A = torch.diag(torch.diag(A))
        A = torch.sqrt(A)
        A_inv = torch.linalg.inv(A)
        V_inv = torch.matmul(A_inv, V_inv)
    return V_inv


def cov_to_corr(cov_matrix: torch.Tensor) -> torch.Tensor:
    """Compute correlation matrix from covariance matrix."""
    S = cov_matrix
    D = torch.sqrt(torch.diag(torch.diag(S)))
    Dinv = torch.linalg.inv(D)
    return torch.linalg.multi_dot([Dinv, S, Dinv])


def calc_rms_ellipse_params(cov_matrix: torch.Tensor) -> tuple[float, ...]:
    """Return projected rms ellipse dimensions and orientation.

    Args:
        cov_matrix: Covariance matrix, shape (2, 2).

    Returns
        c1: Ellipse semi-axis #1.
        c2: Ellipse semi-axis #2.
        angle: Tilt angle below horizontal axis [rad].
    """
    sii = S[0, 0]
    sjj = S[1, 1]
    sij = S[0, 1]
    angle = -0.5 * torch.arctan2(2.0 * sij, sii - sjj)
    _sin = torch.sin(angle)
    _cos = torch.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2
    c1 = torch.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = torch.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))
    return (c1, c2, angle)


class CovFitterBase:
    """Base class for covariance matrix fitting.

    This class uses the Differentiable Evolution global optimization routine.
    """

    def __init__(
        self,
        ndim: int,
        transforms: list[Callable],
        projections: list[list[Histogram]],
        nsamp: int,
        verbose: bool = 2,
        seed: int = None,
        loss_scale: float = 1.0,
        emittance_penalty: float = 0.0,
    ) -> None:
        """Constructor."""
        self.ndim = ndim
        self.nsamp = nsamp
        self.verbose = int(verbose)

        self.params = None
        self.lb = None
        self.ub = None

        self.rng = torch.random.default_rng(seed)
        self.loss_scale = loss_scale
        self.emittance_penalty = emittance_penalty

        self.transforms = transforms
        self.projections = projections

        self.diagnostics = []
        for i in range(len(projections)):
            self.diagnostics.append([proj.copy() for proj in projections[i]])

        self.moments = []
        for proj in unravel(projections):
            if proj.ndim == 1:
                self.moments.append(proj.std() ** 2)
            else:
                _moments = proj.cov()
                _moments = _moments[torch.tril_indices(_moments.ndim)]
                _moments = _moments.tolist()
                self.moments.extend(_moments)
        self.moments = torch.array(self.moments)

        self.iteration = 0
        self.nevals = 0
        self.loss = None
        self.best_loss = torch.inf
        self.best_params = torch.copy(self.params)

    def set_params(self, params: torch.Tensor) -> None:
        """Set covariance matrix parameters."""
        self.params = torch.clip(params, self.lb, self.ub)

    def build_cov(self) -> torch.Tensor:
        """Build covariance matrix from parameters."""
        raise NotImplementedError

    def sample(self, size: int = None) -> torch.Tensor:
        """Sample particles from Gaussian distribution with current covariance matrix."""
        size = size or self.nsamp
        cov_matrix = self.build_cov()
        mean = torch.zeros(self.ndim)
        return self.rng.multivariate_normal(mean, cov_matrix, size=size)

    def simulate(self, x: torch.Tensor) -> torch.Tensor:
        """Track particles and return predicted moments."""
        moments_pred = []
        for index, transform in enumerate(self.transforms):
            x_out = transform(x)
            for diagnostic in self.diagnostics[index]:
                x_out_proj = diagnostic.project(x_out)
                if diagnostic.ndim == 1:
                    moment = torch.var(x_out_proj)
                    moments_pred.append(moment)
                else:
                    cov_matrix = torch.cov(x_out_proj.T)
                    cov_matrix_lt = cov_matrix[torch.tril_indices(cov_matrix.ndim)]
                    cov_matrix_lt = cov_matrix_lt.tolist()
                    moments_pred.extend(cov_matrix_lt)
        return torch.array(moments_pred)

    def loss_function(self, params: torch.Tensor) -> float:
        """Minimizes difference between predicted and measured moments."""
        self.set_params(params)

        x = self.sample()
        y_pred = self.simulate(x)
        y_meas = self.moments

        loss = torch.mean(torch.square(y_pred - y_meas))
        loss = loss * self.loss_scale

        self.loss = loss
        self.nevals += 1

        if self.verbose > 2:
            print(f"loss={self.loss:0.4e} evals={self.nevals}")

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = torch.copy(params)

        return loss

    def fit(
        self, method: str = "differential_evolution", iters: int = 500, **opt_kws
    ) -> tuple[torch.Tensor, OptimizeResult]:
        """Fit parameters to data."""

        def callback_base():
            self.iteration += 1
            if self.verbose > 0:
                print(
                    f"iter={self.iteration:04.0f} loss={self.loss:0.4e} evals={self.nevals}"
                )
            if self.verbose > 1 and self.ndim < 6:
                print(f"cov_matrix:")
                print(self.build_cov())

        result = None

        if method == "simplex":
            opt_kws.setdefault("options", {})
            opt_kws["options"].setdefault("disp", True)
            opt_kws["options"].setdefault("maxiter", iters)

            result = scipy.optimize.minimize(
                self.loss_function,
                self.params,
                method="nelder-mead",
                bounds=scipy.optimize.Bounds(self.lb, self.ub),
                **opt_kws,
            )

        elif method == "powell":
            opt_kws.setdefault("options", {})
            opt_kws["options"].setdefault("disp", True)
            opt_kws["options"].setdefault("maxiter", iters)

            result = scipy.optimize.minimize(
                self.loss_function,
                self.params,
                method="powell",
                bounds=scipy.optimize.Bounds(self.lb, self.ub),
                **opt_kws,
            )

        elif method == "l-bfgs-b":
            opt_kws.setdefault("options", {})
            opt_kws["options"].setdefault("disp", True)
            opt_kws["options"].setdefault("maxiter", iters)

            result = scipy.optimize.minimize(
                self.loss_function,
                self.params,
                method="l-bfgs-b",
                bounds=scipy.optimize.Bounds(self.lb, self.ub),
                **opt_kws,
            )

        elif method == "least_squares":
            opt_kws.setdefault("verbose", 2)
            opt_kws.setdefault("xtol", 1.00e-15)
            opt_kws.setdefault("ftol", 1.00e-15)
            opt_kws.setdefault("gtol", 1.00e-15)
            opt_kws.setdefault("max_nfev", iters)

            result = scipy.optimize.least_squares(
                self.loss_function,
                self.params,
                # bounds=(self.lb, self.ub),
                **opt_kws,
            )

        elif method == "differential_evolution":
            opt_kws.setdefault("popsize", 5)
            opt_kws.setdefault("disp", True)
            opt_kws.setdefault("maxiter", iters)

            result = scipy.optimize.differential_evolution(
                self.loss_function,
                scipy.optimize.Bounds(self.lb, self.ub),
                callback=(lambda intermediate_result: callback_base()),
                **opt_kws,
            )
        elif method == "dual_annealing":
            result = scipy.optimize.dual_annealing(
                self.loss_function,
                scipy.optimize.Bounds(self.lb, self.ub),
                callback=(lambda x, f, context: callback_base()),
                **opt_kws,
            )
        elif method == "shgo":
            result = scipy.optimize.shgo(
                self.loss_function,
                scipy.optimize.Bounds(self.lb, self.ub),
                callback=(lambda x: callback_base()),
                **opt_kws,
            )
        elif method == "direct":
            opt_kws.setdefault("vol_tol", 1.00e-100)
            opt_kws.setdefault("len_tol", 1.00e-18)
            result = scipy.optimize.direct(
                self.loss_function,
                scipy.optimize.Bounds(self.lb, self.ub),
                callback=(lambda x: callback_base()),
                **opt_kws,
            )
        else:
            raise ValueError

        cov_matrix = self.build_cov()
        return cov_matrix, result


class CholeskyCovFitter(CovFitterBase):
    """Parameterizes covariance matrix using Cholesky decomposition S = LL^T."""

    def __init__(self, bound: float = 1.00e15, resample: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        self.nparam = self.ndim * (self.ndim + 1) // 2

        self.L = torch.eye(self.ndim)
        self.z = self.rng.normal(size=(self.nsamp, self.ndim))

        self.idx_diag = (torch.arange(self.ndim), torch.arange(self.ndim))
        self.idx_offdiag = torch.tril_indices(self.ndim, k=-1)

        self.ub = torch.full(self.nparam, bound)
        self.lb = -self.ub
        self.lb[: self.ndim] = 1.00e-15

        self.params = torch.zeros(self.nparam)
        self.params[self.ndim :] = 1.0
        self.set_params(self.params)

    def build_cov(self) -> torch.Tensor:
        self.L[self.idx_diag] = self.params[: self.ndim]
        self.L[self.idx_offdiag] = self.params[self.ndim :]
        return torch.matmul(self.L, self.L.T)

    def set_cov(self, cov_matrix: torch.Tensor) -> None:
        L = torch.linalg.cholesky(cov_matrix)
        self.params[: self.ndim] = L[self.idx_diag]
        self.params[self.ndim :] = L[self.idx_offdiag]

    def set_bounds(self, bound: float) -> None:
        self.ub = torch.full(self.nparam, bound)
        self.lb = -self.ub
        self.lb[: self.ndim] = 1.00e-15

    def sample(self, size: int = None) -> torch.Tensor:
        if size is None:
            size = self.nsamp

        self.L[self.idx_diag] = self.params[: self.ndim]
        self.L[self.idx_offdiag] = self.params[self.ndim :]

        x = self.rng.normal(size=(size, self.ndim))
        x = torch.matmul(x, self.L.T)
        return x


class LinearCovFitter(CovFitterBase):
    """Parameterizes linear transformation of Gaussian base distribution.

    There are N x N parameters with no bounds on the parameters.
    """

    def __init__(self, bound: float = 1.00e15, resample: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nparam = self.ndim**2
        self.ub = torch.full(self.nparam, +bound)
        self.lb = torch.full(self.nparam, -bound)
        self.set_params(torch.ravel(torch.eye(self.ndim)))

    def get_unnorm_matrix(self) -> torch.Tensor:
        return torch.reshape(self.params, (self.ndim, self.ndim))

    def sample(self, size: int = None) -> torch.Tensor:
        size = size or self.nsamp
        unnorm_matrix = self.get_unnorm_matrix()
        x = self.rng.normal(size=(size, self.ndim))
        x = torch.matmul(x, unnorm_matrix.T)
        return x

    def build_cov(self) -> torch.Tensor:
        x = self.sample()
        cov_matrix = torch.cov(x.T)
        return cov_matrix

    def set_cov(self, cov_matrix: torch.Tensor) -> None:
        self.set_params(torch.ravel(torch.linalg.cholesky(cov_matrix)))
