"""Covariance matrix analysis."""
from typing import Callable
from typing import Optional
from typing import Union
from typing import TypeAlias
import numpy as np
import scipy.optimize
import time

from .diag import Histogram1D
from .diag import HistogramND
from .utils import unravel


Histogram: TypeAlias = Union[Histogram1D, HistogramND]  # python<3.12 compatible


def rms_ellipsoid_volume(cov_matrix: np.ndarray) -> float:
    return np.sqrt(np.linalg.det(cov_matrix))


def projected_emittances(cov_matrix: np.ndarray) -> tuple[float, ...]:
    ndim = cov_matrix.shape[0]
    if ndim == 2:
        return rms_ellipsoid_volume(cov_matrix)

    emittances = []
    for i in range(0, cov_matrix.shape[0], 2):
        emittance = rms_ellipsoid_volume(cov_matrix[i : i + 2, i : i + 2])
        emittances.append(emittance)
    return emittances


def intrinsic_emittances(cov_matrix: np.ndarray) -> tuple[float, ...]:
    ndim = cov_matrix.shape[0]
    if ndim > 4:
        raise ValueError("ndim > 4")

    if ndim == 2:
        return rms_ellipsoid_volume(cov_matrix)

    S = cov_matrix.copy()  # [to do] expand to NxN using np.eig
    U = unit_symplectic_matrix(ndim)
    tr_SU2 = np.trace(np.linalg.matrix_power(np.matmul(S, U), 2))
    det_S = np.linalg.det(S)
    eps_1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps_2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return (eps_1, eps_2)


def twiss_2d(cov_matrix: np.ndarray) -> tuple[float, float, float]:
    emittance = rms_ellipsoid_volume(cov_matrix)
    beta = cov_matrix[0, 0] / emittance
    alpha = -cov_matrix[0, 1] / emittance
    return (alpha, beta, emittance)


def twiss(cov_matrix: np.ndarray) -> list[float] | list[list[float]]:
    parameters = []
    for i in range(0, cov_matrix.shape[0], 2):
        parameters.append(twiss_2d(cov_matrix[i : i + 2, i : i + 2]))

    if len(parameters) == 1:
        parameters = parameters[0]

    return parameters


def unit_symplectic_matrix(ndim: int) -> np.ndarray:
    """Return matrix U such that, if M is a symplectic matrix, UMU^T = M."""
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    """Normalize eigenvectors according to Lebedev-Bogacz convention."""
    ndim = eigvecs.shape[0]
    U = unit_symplectic_matrix(ndim)
    for i in range(0, ndim, 2):
        v = eigvecs[:, i]
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0.0:
            (eigvecs[:, i], eigvecs[:, i + 1]) = (eigvecs[:, i + 1], eigvecs[:, i])
        eigvecs[:, i : i + 2] *= np.sqrt(2.0 / np.abs(val))
    return eigvecs


def normalization_matrix_from_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    """Return normalization matrix V^-1 from eigenvectors."""
    V = np.zeros(eigvecs.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigvecs[:, i].real
        V[:, i + 1] = (1.0j * eigvecs[:, i]).real
    return np.linalg.inv(V)


def normalization_matrix_from_twiss_2d(
    alpha: float, beta: float, emittance: float = None
) -> np.ndarray:
    """Return 2 x 2 normalization matrix V^-1 from Twiss parameters."""
    V = np.array([[beta, 0.0], [-alpha, 1.0]]) * np.sqrt(1.0 / beta)
    A = np.eye(2)
    if emittance is not None:
        A = np.sqrt(np.diag([emittance, emittance]))
    V = np.matmul(V, A)
    return np.linalg.inv(V)


def normalization_matrix_from_twiss(
    twiss_params: list[tuple[float, float, float]]
) -> np.ndarray:
    """2N x 2N block-diagonal normalization matrix from Twiss parameters.

    Parameters
    ----------
    twiss_params : list[tuple[float, float, float]]
        Twiss parameters (alpha, beta, emittance) in each dimension.

    Returns
    -------
    V : ndarray, shape (2N, 2N)
        Block-diagonal normalization matrix.
    """
    ndim = len(twiss_params) // 2
    V = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        V[i : i + 2, i : i + 2] = normalization_matrix_from_twiss_2d(
            *twiss_params[i : i + 2]
        )
    return np.linalg.inv(V)


def normalization_matrix(
    cov_matrix: np.ndarray, scale: bool = False, block_diag: bool = False
) -> np.ndarray:
    """Return normalization matrix V^{-1} from covariance matrix S.

    Parameters
    ----------
    cov_matrix : np.ndarray
        An N x N covariance matrix.
    scale : bool
        If True, normalize to unit rms emittance.
    block_diag : bool
        If true, normalize only 2x2 block-diagonal elements (x-x', y-y', etc.).
    """
    def _normalization_matrix(cov_matrix: np.ndarray, scale: bool = False) -> np.ndarray:
        S = cov_matrix.copy()
        U = unit_symplectic_matrix(S.shape[0])
        eigvals, eigvecs = np.linalg.eig(np.matmul(S, U))
        eigvecs = normalize_eigvecs(eigvecs)
        V_inv = normalization_matrix_from_eigvecs(eigvecs)

        if scale:
            ndim = S.shape[0]
            V = np.linalg.inv(V_inv)
            A = np.eye(ndim)
            if ndim == 2:
                emittance = np.sqrt(np.linalg.det(S))
                A = np.diag(np.sqrt([emittance, emittance]))
            else:
                S_n = np.linalg.multi_dot([V_inv, S, V_inv.T])
                A = np.sqrt(np.diag(np.repeat(projected_emittances(S_n), 2)))
            V = np.matmul(V, A)
            V_inv = np.linalg.inv(V)

        return V_inv

    ndim = cov_matrix.shape[0]
    norm_matrix = np.eye(ndim)
    if block_diag:
        for i in range(0, ndim, 2):
            norm_matrix[i: i + 2, i: i + 2] = _normalization_matrix(
                cov_matrix[i: i + 2, i: i + 2], scale=scale
            )
    else:
        norm_matrix = _normalization_matrix(cov_matrix, scale=scale)
    return norm_matrix


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(S.diagonal()))
    Dinv = np.linalg.inv(D)
    return np.linalg.multi_dot([Dinv, S, Dinv])


def rms_ellipse_params(
    S: np.ndarray, axis: tuple[int, ...] = None
) -> tuple[float, ...]:
    """Return projected rms ellipse dimensions and orientation.

    Parameters
    ----------
    S : ndarray, shape (2N, 2N)
        The phase space covariance matrix.
    axis : tuple[int]
        Projection axis. Example: if the axes are {x, xp, y, yp}, and axis=(0, 2),
        the four-dimensional ellipsoid is projected onto the x-y plane.

    Returns
    -------
    c1, c2 : float
        The ellipse semi-axis widths.
    angle : float
        The tilt angle below the x axis [radians].
    """
    if S.shape[0] == 2:
        axis = (0, 1)
    (i, j) = axis
    sii = S[i, i]
    sjj = S[j, j]
    sij = S[i, j]
    angle = -0.5 * np.arctan2(2 * sij, sii - sjj)
    _sin = np.sin(angle)
    _cos = np.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2
    c1 = np.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = np.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))
    return (c1, c2, angle)



class CovFitterBase:
    """Base class for covariance matrix fitting classes.

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
    ) -> None:
        """Constructor."""
        self.ndim = ndim
        self.nsamp = nsamp
        self.verbose = verbose
        
        self.params = None
        self.rng = np.random.default_rng(seed)
        self.loss_scale = loss_scale

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
                _moments = _moments[np.tril_indices(_moments.ndim)]
                _moments = _moments.tolist()
                self.moments.extend(_moments)
        self.moments = np.array(self.moments)
        
    def set_params(self, params: np.ndarray) -> None:
        """Set covariance matrix parameters."""
        self.params = params
                
    def build_cov(self) -> np.ndarray:
        """Build covariance matrix from parameters."""
        raise NotImplementedError

    def sample(self, size: int = None) -> np.ndarray:        
        """Sample particles from Gaussian distribution with current covariance matrix."""
        if size is None:
            size = self.nsamp
        
        cov = self.build_cov()
        mean = np.zeros(self.ndim)
        return self.rng.multivariate_normal(mean, cov, size=size)

    def simulate(self, x: np.ndarray) -> np.ndarray:
        """Track particles and return predicted moments."""
        moments_pred = []
        for index, transform in enumerate(self.transforms):
            x_out = transform(x)
            for diagnostic in self.diagnostics[index]:
                x_out_proj = diagnostic.project(x_out)
                if diagnostic.ndim == 1:
                    moment = np.var(x_out_proj)
                    moments_pred.append(moment)
                else:
                    cov_matrix = np.cov(x_out_proj.T)
                    cov_matrix_lt = cov_matrix[np.tril_indices(cov_matrix.ndim)]
                    cov_matrix_lt = cov_matrix_lt.tolist()
                    moments_pred.extend(cov_matrix_lt)
        return np.array(moments_pred)
        
    def loss_function(self, params: np.ndarray) -> np.ndarray:
        """Minimizes difference between predicted and measured moments."""
        self.set_params(params)
        y_pred = self.simulate(self.sample())
        y_meas = self.moments
        loss = np.mean(np.square(y_pred - y_meas))     
        loss = loss * self.loss_scale
        return loss
        
    def get_param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds on parameters."""
        raise NotImplementedError

    def fit(self, **opt_kws) -> tuple[np.ndarray, scipy.optimize.OptimizeResult]:
        """Fit parameters to data."""
        
        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            self.set_params(intermediate_result.x)
            cov_matrix = self.build_cov()
            print("cov_matrix:")
            print(cov_matrix)

        def is_semi_positive_definite(matrix: np.ndarray) -> bool:
            if not np.array_equal(matrix, matrix.T):
                return False             
            return np.all(np.linalg.eigvals(matrix) >= 0.0)

        def constraint(params: np.ndarray) -> np.ndarray:
            return int(is_semi_positive_definite(self.build_cov()))
            
        constraints = [
            scipy.optimize.NonlinearConstraint(constraint, 0.0, np.inf),
        ]

        if self.verbose:
            opt_kws.setdefault("disp", True)
            opt_kws.setdefault("callback", callback)

        result = scipy.optimize.differential_evolution(
            self.loss_function, 
            self.get_param_bounds(), 
            # constraints=constraints,
            **opt_kws
        )
        
        cov_matrix = self.build_cov()
        return cov_matrix, result


class CholeskyCovFitter(CovFitterBase):
    """Parameterizes covariance matrix using Cholesky decomposition S = LL^T.""" 
    def __init__(self, bound: float = 1.00e+15, resample: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.nparam = self.ndim * (self.ndim + 1) // 2
        self.params = np.zeros(self.nparam)
        self.params[self.ndim:] = 1.0

        self.L = np.eye(self.ndim)
        self.z = self.rng.normal(size=(self.nsamp, self.ndim))
        self.resample = resample

        self.idx_diag = (np.arange(self.ndim), np.arange(self.ndim))
        self.idx_offdiag = np.tril_indices(self.ndim, k=-1)

        self.ub = np.full(self.nparam, bound)
        self.lb = -self.ub
        self.lb[:self.ndim] = 1.00e-15

    def build_cov(self) -> np.ndarray:
        self.L[self.idx_diag] = self.params[:self.ndim]
        self.L[self.idx_offdiag] = self.params[self.ndim:]
        return np.matmul(self.L, self.L.T)

    def set_cov(self, cov_matrix: np.ndarray) -> None:
        L = np.linalg.cholesky(cov_matrix)
        self.params[:self.ndim] = L[self.idx_diag]
        self.params[self.ndim:] = L[self.idx_offdiag]

    def set_bounds(self, bound: float) -> None:
        self.ub = np.full(self.nparam, bound)
        self.lb = -self.ub
        self.lb[:self.ndim] = 1.00e-15
        
    def get_param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return scipy.optimize.Bounds(self.lb, self.ub)

    def sample(self, size: int = None) -> np.ndarray:        
        if size is None:
            size = self.nsamp

        self.L[self.idx_diag] = self.params[:self.ndim]
        self.L[self.idx_offdiag] = self.params[self.ndim:]
        # x = self.rng.normal(size=(size, self.ndim))
        # x = np.matmul(x, self.L.T)

        x = np.matmul(self.z, self.L.T)
        return x





    