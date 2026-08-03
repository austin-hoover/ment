"""Covariance matrix fitting."""
from collections.abc import Callable

import numpy as np
import torch
import scipy.optimize
from scipy.optimize import OptimizeResult
from scipy.optimize import Bounds

from ..diag import Histogram
from ..utils import array_to_tensor


class CovFitterBase:
    def __init__(
        self,
        ndim: int,
        transforms: list[Callable],
        projections: list[list[Histogram]],
        nsamp: int,
        verbose: bool = 2,
        loss_scale: float = 1.0,
        emittance_penalty: float = 0.0,
    ) -> None:

        self.ndim = ndim
        self.nsamp = nsamp
        self.verbose = int(verbose)

        self.params = None
        self.lb = None
        self.ub = None

        self.loss_scale = loss_scale
        self.emittance_penalty = emittance_penalty

        self.transforms = transforms
        self.projections = projections

        self.diagnostics = []
        for i in range(len(projections)):
            self.diagnostics.append([p.copy() for p in projections[i]])

        self.iteration = 0
        self.nevals = 0
        self.loss = None
        self.best_loss = np.inf
        self.best_params = None

    def set_params(self, params: np.ndarray) -> None:
        self.params = np.clip(params, self.lb, self.ub)

    def build_cov(self) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, size: int = None) -> torch.Tensor:
        size = size or self.nsamp
        cov_matrix = self.build_cov().float()
        L = torch.linalg.cholesky(cov_matrix)
        x = torch.randn((size, self.ndim))
        return x @ L.T

    def loss_function(self, params: np.ndarray) -> float:
        self.set_params(params)
        x = self.sample()

        loss = 0.0
        for i, transform in enumerate(self.transforms):
            x_out = transform(x)
            for j, diagnostic in enumerate(self.diagnostics[i]):
                x_out_proj = torch.squeeze(diagnostic.project(x_out))
                if x_out_proj.ndim == 1:
                    var_pred = torch.var(x_out_proj)
                    var_meas = self.projections[i][j].var()
                    loss += float(torch.abs(var_pred - var_meas))
                else:
                    cov_pred = torch.cov(x_out_proj.T)
                    cov_meas = self.projections[i][j].cov()
                    loss += float(torch.mean(torch.abs(cov_pred - cov_meas)))

        loss = loss / (i + 1)
        loss = loss * self.loss_scale
        self.loss = loss
        self.nevals += 1

        if self.verbose > 2:
            print(f"loss={self.loss:0.4e} evals={self.nevals}")

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = torch.clone(torch.as_tensor(params))

        return loss

    def fit(
        self, method: str = "differential_evolution", iters: int = 500, **opt_kws
    ) -> tuple[np.ndarray, OptimizeResult]:
        def callback_base():
            self.iteration += 1
            if self.verbose > 0:
                print(
                    f"iter={self.iteration:04.0f} loss={self.loss:0.4e} evals={self.nevals}"
                )
            if self.verbose > 1 and self.ndim < 6:
                print(f"cov_matrix:")
                print(self.build_cov())

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

        self.idx_diag = (np.arange(self.ndim), np.arange(self.ndim))
        self.idx_offdiag = np.tril_indices(self.ndim, k=-1)

        self.ub = np.full(self.nparam, bound)
        self.lb = -self.ub
        self.lb[: self.ndim] = 1.00e-15

        self.params = np.zeros(self.nparam)
        self.params[self.ndim :] = 1.0
        self.set_params(self.params)

    def build_cov(self) -> np.array:
        self.L[self.idx_diag] = array_to_tensor(self.params[: self.ndim])
        self.L[self.idx_offdiag] = array_to_tensor(self.params[self.ndim :])
        return np.matmul(self.L, self.L.T)

    def set_cov(self, cov_matrix: torch.Tensor) -> None:
        L = torch.linalg.cholesky(cov_matrix)
        self.params[: self.ndim] = L[self.idx_diag].numpy()
        self.params[self.ndim :] = L[self.idx_offdiag].numpy()

    def set_bounds(self, bound: float) -> None:
        self.ub = np.full(self.nparam, bound)
        self.lb = -self.ub
        self.lb[: self.ndim] = 1.00e-15

    def sample(self, size: int = None) -> torch.Tensor:
        size = size or self.nsamp

        self.L[self.idx_diag] = array_to_tensor(self.params[: self.ndim])
        self.L[self.idx_offdiag] = array_to_tensor(self.params[self.ndim :])

        x = torch.randn((size, self.ndim))
        return x @ self.L.T


class LinearCovFitter(CovFitterBase):
    """Parameterizes linear transformation of Gaussian base distribution.

    There are N x N parameters with no bounds.
    """

    def __init__(self, bound: float = 1.00e15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nparam = self.ndim**2
        self.ub = np.full(self.nparam, +bound)
        self.lb = np.full(self.nparam, -bound)
        self.set_params(np.ravel(np.eye(self.ndim)))

    def get_unnorm_matrix(self) -> torch.Tensor:
        matrix = np.reshape(self.params, (self.ndim, self.ndim))
        matrix = torch.from_numpy(matrix).float()
        return matrix

    def sample(self, size: int = None) -> np.ndarray:
        size = size or self.nsamp
        matrix = self.get_unnorm_matrix()
        x = torch.randn((size, self.ndim))
        return x @ matrix.T

    def build_cov(self) -> torch.Tensor:
        x = self.sample()
        return torch.cov(x.T)

    def set_cov(self, cov_matrix: torch.Tensor) -> None:
        self.set_params(torch.ravel(torch.linalg.cholesky(cov_matrix)))
