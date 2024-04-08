import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


class LinearTransform:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.matrix.T)