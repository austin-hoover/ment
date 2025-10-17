import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.utils.consts import speed_of_light

from ment.cov import norm_matrix


def get_part_coords(bunch: Bunch, index: int) -> list[float]:
    x = bunch.x(index)
    y = bunch.y(index)
    z = bunch.z(index)
    xp = bunch.xp(index)
    yp = bunch.yp(index)
    de = bunch.dE(index)
    return [x, xp, y, yp, z, de]


def set_part_coords(bunch: Bunch, index: int, coords: list[float]) -> Bunch:
    (x, xp, y, yp, z, de) = coords
    bunch.x(index, x)
    bunch.y(index, y)
    bunch.z(index, z)
    bunch.xp(index, xp)
    bunch.yp(index, yp)
    bunch.dE(index, de)
    return bunch


def get_bunch_coords(bunch: Bunch, axis: tuple[int, ...] = None) -> np.ndarray:
    if axis is None:
        axis = tuple(range(6))

    X = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        X[i, :] = get_part_coords(bunch, i)
    return X[:, axis]


def set_bunch_coords(
    bunch: Bunch, X: np.ndarray, axis: tuple[int, ...] = None
) -> Bunch:
    if axis is None:
        axis = tuple(range(6))

    # Resize the bunch if necessary
    size = X.shape[0]
    size_error = size - bunch.getSize()
    if size_error > 0:
        for _ in range(size_error):
            bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        for i in range(size, bunch.getSize()):
            bunch.deleteParticleFast(i)
        bunch.compress()

    for index in range(bunch.getSize()):
        coords = get_part_coords(bunch, index)
        for j, _axis in enumerate(axis):
            coords[_axis] = X[index, j]
        bunch = set_part_coords(bunch, index, coords)
    return bunch


def reverse_bunch(bunch: Bunch) -> Bunch:
    size = bunch.getSize()
    for i in range(size):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
    return bunch


def get_bunch_centroid(bunch: Bunch) -> np.ndarray:
    calc = BunchTwissAnalysis()
    calc.analyzeBunch(bunch)
    return np.array([calc.getAverage(i) for i in range(6)])


def set_bunch_centroid(bunch: Bunch, centroid: np.ndarray) -> Bunch:
    centroid_shift = centroid - get_bunch_centroid(bunch)
    bunch = shift_bunch_centroid(bunch, *centroid_shift)
    return bunch


def set_bunch_cov(bunch: Bunch, S: np.ndarray, block_diag: bool = True) -> Bunch:
    X_old = get_bunch_coords(bunch)
    S_old = np.cov(X_old.T)

    # Assume block-diagonal covariance matrix
    V_old_inv = norm_matrix(S_old, scale=True, block_diag=block_diag)
    V_old = np.linalg.inv(V_old_inv)

    S_new = S.copy()
    V_new_inv = norm_matrix(S_new, scale=True, block_diag=block_diag)
    V_new = np.linalg.inv(V_new_inv)

    M = np.matmul(V_new, V_old_inv)
    X_new = np.matmul(X_old, M.T)

    bunch = set_bunch_coords(bunch, X_new)
    return bunch


def shift_bunch_centroid(
    bunch: Bunch,
    x: float = 0.0,
    xp: float = 0.0,
    y: float = 0.0,
    yp: float = 0.0,
    z: float = 0.0,
    de: float = 0.0,
) -> Bunch:
    for i in range(bunch.getSize()):
        bunch.x(i, bunch.x(i) + x)
        bunch.y(i, bunch.y(i) + y)
        bunch.z(i, bunch.z(i) + z)
        bunch.xp(i, bunch.xp(i) + xp)
        bunch.yp(i, bunch.yp(i) + yp)
        bunch.dE(i, bunch.dE(i) + de)
    return bunch


def get_bunch_coords(bunch: Bunch) -> np.ndarray:
    X = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        X[i, 0] = bunch.x(i)
        X[i, 1] = bunch.xp(i)
        X[i, 2] = bunch.y(i)
        X[i, 3] = bunch.yp(i)
        X[i, 4] = bunch.z(i)
        X[i, 5] = bunch.dE(i)
    return X


def set_bunch_coords(bunch: Bunch, X: np.ndarray) -> Bunch:
    bunch.deleteAllParticles()
    bunch.compress()
    for i in range(X.shape[0]):
        x, xp, y, yp, z, de = X[i, :]
        bunch.addParticle(x, xp, y, yp, z, de)
    return bunch


def transform_bunch(bunch: Bunch, M: np.array) -> Bunch:
    X = get_bunch_coords(bunch)
    X = np.matmul(X, M.T)
    for i in range(X.shape[0]):
        bunch.x(i, X[i, 0])
        bunch.y(i, X[i, 2])
        bunch.z(i, X[i, 4])
        bunch.xp(i, X[i, 1])
        bunch.yp(i, X[i, 3])
        bunch.dE(i, X[i, 5])
    return bunch


def get_z_to_phase_coefficient(bunch: Bunch, frequency: float) -> float:
    """Return coefficient to calculate rf phase [degrees] from z [m]."""
    velocity = bunch.getSyncParticle().beta() * speed_of_light
    wavelength = velocity / frequency
    coefficient = 360.0 / wavelength
    return coefficient
