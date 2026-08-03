"""Covariance matrix analysis."""
import numpy as np
import torch


def normalize_eigvec(v: torch.Tensor) -> torch.Tensor:
    """Normalize eigenvectors according to Lebedev-Bogacz convention.

    conj(v)^T U v = 2i
    """
    ndim = len(v)
    v = torch.clone(torch.as_tensor(v))
    U = build_poisson_matrix(ndim=ndim, complex=True).to(device=v.device)

    def norm(vec: torch.Tensor) -> torch.Tensor:
        return torch.linalg.multi_dot([torch.conj(v), U, v])

    if torch.imag(norm(v)) > 0:
        v = torch.conj(v)

    v *= torch.sqrt(2.0 / torch.abs(norm(v)))
    assert torch.isclose(torch.imag(norm(v)), torch.tensor(-2.0, device=v.device))
    assert torch.isclose(torch.real(norm(v)), torch.tensor(0.0, device=v.device))
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
    sii = cov_matrix[0, 0]
    sjj = cov_matrix[1, 1]
    sij = cov_matrix[0, 1]

    angle = -0.5 * torch.arctan2(2.0 * sij, sii - sjj)

    _sin = torch.sin(angle)
    _cos = torch.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2

    c1 = torch.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = torch.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))
    return (c1, c2, angle)
