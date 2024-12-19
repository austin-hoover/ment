import numpy as np


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


def norm_matrix_from_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    """Return normalization matrix V^-1 from eigenvectors."""
    V = np.zeros(eigvecs.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigvecs[:, i].real
        V[:, i + 1] = (1.0j * eigvecs[:, i]).real
    return np.linalg.inv(V)


def norm_matrix(S: np.ndarray, scale: bool = False, block_diag: bool = False) -> np.ndarray:
    """Return normalization matrix V^{-1} from covariance matrix S.

    Parameters
    ----------
    S : np.ndarray
        An N x N covariance matrix.
    scale : bool
        If True, normalize to unit rms emittance.
    block_diag : bool
        If true, normalize only 2x2 block-diagonal elements (x-x', y-y', etc.).
    """
    ndim = S.shape[0]
    V_inv = np.eye(ndim)
    if block_diag:
        V_inv = _norm_matrix(S, scale=scale)
    else:
        V_inv = _norm_matrix_block_diag(S, scale=scale)
    return V_inv


def _norm_matrix(S: np.ndarray, scale: bool = False) -> np.ndarray:
    ndim = S.shape[0]
    assert ndim % 2 == 0

    U = unit_symplectic_matrix(ndim)
    SU = np.matmul(S, U)
    eigvals, eigvecs = np.linalg.eig(SU)
    eigvecs = normalize_eigvecs(eigvecs)
    V_inv = norm_matrix_from_eigvecs(eigvecs)
    if scale:
        V = np.linalg.inv(V_inv)
        A = np.eye(ndim)
        for i in range(0, ndim, 2):
            emittance = np.sqrt(np.linalg.det(S[i : i + 2, i : i + 2]))
            A[i : i + 2, i : i + 2] *= np.sqrt(emittance)
        V = np.matmul(V, A)
        V_inv = np.linalg.inv(V)
    return V_inv


def _norm_matrix_block_diag(S: np.ndarray, scale: bool = False) -> np.ndarray:
    ndim = S.shape[0]
    V_inv = np.eye(ndim)
    for i in range(0, ndim, 2):
        V_inv[i : i + 2, i : i + 2] = _norm_matrix(S[i : i + 2, i : i + 2], scale=scale)
    return V_inv
