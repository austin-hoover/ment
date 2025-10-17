"""Covariance matrix analysis and fitting routines."""
import numpy as np


# General
# --------------------------------------------------------------------------------------


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


def normalization_matrix(
    S: np.ndarray, scale: bool = False, block_diag: bool = False
) -> np.ndarray:
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

    def _normalization_matrix(S: np.ndarray, scale: bool = False) -> np.ndarray:
        S = np.copy(S)
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
                A = np.sqrt(np.diag(np.repeat(compute_projected_emittances(S_n), 2)))
            V_inv = np.linalg.inv(np.matmul(V, A))
        return V_inv

    ndim = S.shape[0]
    V_inv = np.eye(ndim)
    if block_diag:
        for i in range(0, ndim, 2):
            V_inv[i : i + 2, i : i + 2] = _normalization_matrix(
                S[i : i + 2, i : i + 2], scale=scale
            )
    else:
        V_inv = _normalization_matrix(S, scale=scale)
    return V_inv


def compute_emittance(S: np.ndarray) -> float:
    """Return ND emittance from covariance matrix."""
    return np.sqrt(np.linalg.det(S))


def compute_projected_emittances(S: np.ndarray) -> tuple[float, float]:
    """Compute projected emittances epsx, epsy, ... from covariance matrix."""
    emittances = []
    for i in range(0, S.shape[0], 2):
        emittance = compute_emittance(S[i : i + 2, i : i + 2])
        emittances.append(emittance)
    return emittances


def compute_intrinsic_emittances(S: np.ndarray) -> tuple[float, float]:
    """Compute intrinsic emittances eps1, eps2, ... from covariance matrix."""
    assert S.shape[0] == S.shape[1] == 4

    S = np.copy(S[:4, :4])
    U = unit_symplectic_matrix(4)
    SU = np.matmul(S, U)
    SU2 = np.linalg.matrix_power(SU, 2)
    tr_SU2 = np.trace(SU2)
    det_S = np.linalg.det(S)
    eps_1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps_2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return (eps_1, eps_2)


def compute_twiss(S: np.ndarray) -> list[float] | list[list[float]]:
    """Compute twiss parameters [(alpha_x, beta_x), (alpha_y, beta_y), ...] from covariance matrix."""
    parameters = []
    for i in range(0, S.shape[0], 2):
        emittance = compute_emittance(S[i : i + 2, i : i + 2])
        alpha = -S[i, i + 1] / emittance
        beta = S[i, i] / emittance
        parameters.append([alpha, beta])

    if len(parameters) == 1:
        parameters = parameters[0]

    return parameters


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(S.diagonal()))
    Dinv = np.linalg.inv(D)
    return np.linalg.multi_dot([Dinv, S, Dinv])


def rms_ellipse_params(S: np.ndarray) -> tuple[float, float, float]:
    """Return rms ellipse dimensions and orientation.

    Parameters
    ----------
    S : ndarray, shape (2, 2)
        A two-dimensional covariance matrix.

    Returns
    -------
    c1, c2 : float
        The ellipse semi-axis widths.
    angle : float
        The tilt angle below the x axis [radians].
    """
    (i, j) = (0, 1)

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


# Fitting
# --------------------------------------------------------------------------------------


def cov_vec_to_mat(vec: np.ndarray) -> np.ndarray:
    (S11, S22, S12, S33, S44, S34, S13, S23, S14, S24) = vec
    S = [
        [S11, S12, S13, S14],
        [S12, S22, S23, S24],
        [S13, S23, S33, S34],
        [S14, S24, S34, S44],
    ]
    S = np.array(S)
    return S


def cov_mat_to_vec(S: np.ndarray) -> np.ndarray:
    S11, S12, S13, S14 = S[0, :]
    S22, S23, S24 = S[1, 1:]
    S33, S34 = S[2, 2:]
    S44 = S[3, 3]
    return np.array([S11, S22, S12, S33, S44, S34, S13, S23, S14, S24])


def compute_cov_xy(cov_xx: float, cov_yy: float, cov_uu: float):
    cov_xy = cov_uu - 0.5 * (cov_xx + cov_yy)
    cov_xy *= -1.0
    return cov_xy


def compute_cov_uu(cov_xx: float, cov_yy: float, cov_xy: float):
    return 0.5 * (cov_xx + cov_yy) - cov_xy


def fit_cov(moments: np.ndarray, tmats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the covariance matrix.

    Parameters
    ----------
    moments : list or ndarray, shape (n, 3)
        The [<xx>, <yy>, <xy>] moments.
    tmats: list or ndarray, shape (n, 4, 4)
        Transfer matrices from the reconstruction location to the measurement locations.

    Returns
    -------
    S : ndarray, shape (4, 4)
        The reconstructed covariance matrix.
    C : ndarray, shape (10, 10)
        The LLSQ covariance matrix.
    """
    # Form coefficient arrays and target arrays,
    Axx, Ayy, Axy, bxx, byy, bxy = [], [], [], [], [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(tmats, moments):
        Axx.append([M[0, 0] ** 2, M[0, 1] ** 2, 2 * M[0, 0] * M[0, 1]])
        Ayy.append([M[2, 2] ** 2, M[2, 3] ** 2, 2 * M[2, 2] * M[2, 3]])
        Axy.append(
            [M[0, 0] * M[2, 2], M[0, 1] * M[2, 2], M[0, 0] * M[2, 3], M[0, 1] * M[2, 3]]
        )
        bxx.append(sig_xx)
        byy.append(sig_yy)
        bxy.append(sig_xy)
    Axx = np.array(Axx)
    Ayy = np.array(Ayy)
    Axy = np.array(Axy)

    # Solve LLSQ problem.
    vec_xx, res_xx, _, _ = np.linalg.lstsq(Axx, bxx, rcond=None)
    vec_yy, res_yy, _, _ = np.linalg.lstsq(Ayy, byy, rcond=None)
    vec_xy, res_xy, _, _ = np.linalg.lstsq(Axy, bxy, rcond=None)

    # Form beam covariance matrix.
    vec = np.hstack([vec_xx, vec_yy, vec_xy])
    S = cov_vec_to_mat(vec)

    # Estimate standard deviation of fit parameters.

    if len(res_xy) == 0:
        res_xy = 1.00e-08

    Cxx = llsq_variance_covariance_matrix(Axx, float(res_xx))
    Cyy = llsq_variance_covariance_matrix(Ayy, float(res_yy))
    Cxy = llsq_variance_covariance_matrix(Axy, float(res_xy))

    C = np.zeros((10, 10))
    C[0:3, 0:3] = Cxx
    C[3:6, 3:6] = Cyy
    C[6:, 6:] = Cxy

    return S, C


# LLSQ error propagation. Should check these again. Also revisit with weighted
# least squares.
# --------------------------------------------------------------------------------------


def compute_projected_emittance_std(S: np.ndarray, C: np.ndarray) -> float:
    eps = compute_emittance(S)
    grad_eps = (0.5 / eps) * np.array([S[1, 1], S[0, 0], -2.0 * S[0, 1]])
    eps_std = np.sqrt(np.linalg.multi_dot([grad_eps.T, C, grad_eps]))
    return eps_std


def compute_projected_emittance_stds(
    S: np.ndarray, C: np.ndarray
) -> tuple[float, float]:
    eps_x_std = compute_projected_emittance_std(S[0:2, 0:2], C[0:3, 0:3])
    eps_y_std = compute_projected_emittance_std(S[2:4, 2:4], C[3:6, 3:6])
    return (eps_x_std, eps_y_std)


def compute_intrinsic_emittance_stds(
    S: np.ndarray, C: np.ndarray
) -> tuple[float, float]:
    eps_1, eps_2 = compute_intrinsic_emittances(S)

    Cxx = C[0:3, 0:3]
    Cyy = C[3:6, 3:6]
    Cxy = C[6:, 6:]

    U = unit_symplectic_matrix(4)
    SU = np.matmul(S, U)

    g1 = np.linalg.det(S)
    g2 = np.trace(np.matmul(SU, SU))
    g = np.array(g1, g2)

    S11, S12, S13, S14 = S[0, :]
    S22, S23, S24 = S[1, 1:]
    S33, S34 = S[2, 2:]
    S44 = S[3, 3]

    grad_g = np.zeros((10, 2))

    grad_g[0, 0] = (
        -S33 * S24**2
        + 2.0 * S23 * S24 * S34
        - S44 * S23**2
        + S22 * (S33 * S44 - S34**2)
    )
    grad_g[1, 0] = (
        -S33 * S14**2
        + 2.0 * S13 * S14 * S34
        - S44 * S13**2
        + S11 * (S33 * S44 - S34**2)
    )
    grad_g[2, 0] = 2.0 * (
        S14 * (S24 * S33 - S23 * S34)
        + S13 * (-S24 * S34 + S23 * S44)
        + S12 * (S34**2 - S33 * S44)
    )
    grad_g[3, 0] = (
        -S22 * S14**2
        + 2.0 * S12 * S14 * S24
        - S44 * S12**2
        + S11 * (S22 * S44 - S24**2)
    )
    grad_g[4, 0] = (
        -S22 * S13**2
        + 2.0 * S12 * S13 * S23
        - S33 * S12**2
        + S11 * (S22 * S33 - S23**2)
    )
    grad_g[5, 0] = 2.0 * (
        -S12 * S14 * S23
        + S13 * (S14 * S22 - S12 * S24)
        + S34 * S12**2
        + S11 * (S23 * S24 - S22 * S34)
    )
    grad_g[6, 0] = 2.0 * (
        S14 * (-S23 * S24 + S22 * S34)
        + S13 * (S24**2 - S22 * S44)
        + S12 * (-S24 * S34 + S23 * S44)
    )
    grad_g[7, 0] = 2.0 * (
        S23 * S14**2
        - S14 * (S13 * S24 + S12 * S34)
        + S12 * S13 * S44
        + S11 * (S24 * S34 - S23 * S44)
    )
    grad_g[8, 0] = 2.0 * (
        S14 * (S23**2 - S22 * S33)
        + S13 * (-S23 * S24 + S22 * S34)
        + S12 * (S24 * S33 - S23 * S34)
    )
    grad_g[9, 0] = 2.0 * (
        S24 * S13**2
        + S12 * S14 * S33
        - S13 * (S14 * S23 + S12 * S34)
        + S11 * (-S24 * S33 + S23 * S34)
    )

    grad_g[0, 1] = -2.0 * S22
    grad_g[1, 1] = -2.0 * S11
    grad_g[2, 1] = +4.0 * S12
    grad_g[3, 1] = -2.0 * S44
    grad_g[4, 1] = -2.0 * S33
    grad_g[5, 1] = +4.0 * S34
    grad_g[6, 1] = -4.0 * S24
    grad_g[7, 1] = +4.0 * S24
    grad_g[8, 1] = +4.0 * S23
    grad_g[9, 1] = -4.0 * S13

    Cg = np.linalg.multi_dot([grad_g.T, C, grad_g])

    H = np.sqrt(g2**2 - 16.0 * g1)
    deps1_dg1 = -1.0 / (eps_1 * H)
    deps2_dg1 = -1.0 / (eps_2 * H)
    deps1_dg2 = (1.0 / 8.0) * (1.0 / eps_1) * (+g2 / H - 1.0)
    deps2_dg2 = (1.0 / 8.0) * (1.0 / eps_2) * (-g2 / H - 1.0)
    grad_eps_1 = np.array([deps1_dg1, deps1_dg2])
    grad_eps_2 = np.array([deps2_dg1, deps2_dg2])

    eps_1_std = np.linalg.multi_dot([grad_eps_1.T, Cg, grad_eps_1])
    eps_2_std = np.linalg.multi_dot([grad_eps_2.T, Cg, grad_eps_2])

    return (eps_1_std, eps_2_std)


def compute_beta_std(S: np.ndarray, C: np.ndarray) -> float:
    eps = compute_emittance(S)
    alpha, beta = compute_twiss(S)

    a = S[0, 0]
    b = S[1, 1]
    c = S[0, 1]

    grad_beta = np.zeros(3)
    grad_beta[0] = (1.0 / eps) - ((a * b) / (2.0 * eps**3))
    grad_beta[1] = -(a**2) / (2.0 * eps**2)
    grad_beta[2] = (a * c) / (eps**3)

    beta_std = np.sqrt(np.linalg.multi_dot([grad_beta.T, C, grad_beta]))
    return beta_std


def compute_alpha_std(S: np.ndarray, C: np.ndarray) -> float:
    eps = compute_emittance(S)
    alpha, beta = compute_twiss(S)

    a = S[0, 0]
    b = S[1, 1]
    c = S[0, 1]

    grad_alpha = np.zeros(3)
    grad_alpha[0] = (c * b) / (2.0 * eps**3)
    grad_alpha[1] = (c * a) / (2.0 * eps**3)
    grad_alpha[2] = -(1.0 / eps) - ((c**2) / (eps**3))

    alpha_std = np.sqrt(np.linalg.multi_dot([grad_alpha.T, C, grad_alpha]))
    return alpha_std


def compute_twiss_stds_2x2(S: np.ndarray, C: np.ndarray) -> tuple[float, float]:
    alpha_std = compute_alpha_std(S, C)
    beta_std = compute_beta_std(S, C)
    return (alpha_std, beta_std)


def compute_twiss_stds(
    S: np.ndarray, C: np.ndarray
) -> tuple[float, float, float, float]:
    (alpha_x_std, beta_x_std) = compute_twiss_stds_2x2(S[0:2, 0:2], C[0:3, 0:3])
    (alpha_y_std, beta_y_std) = compute_twiss_stds_2x2(S[2:4, 2:4], C[3:6, 3:6])
    return (alpha_x_std, beta_x_std, alpha_y_std, beta_y_std)


def llsq_variance_covariance_matrix(coefficient_matrix, sum_of_squared_residuals):
    A = coefficient_matrix
    H = np.matmul(A.T, A)
    (n, m) = np.shape(A)
    if n == m:
        C = np.linalg.inv(A)
    else:
        C = (sum_of_squared_residuals / (n - m)) * np.linalg.inv(H)
    return C
