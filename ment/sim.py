import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


class LinearTransform:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.matrix_inv = np.linalg.inv(matrix)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.matrix.T)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.matrix_inv.T)


def forward(x, transforms, diagnostics):
    predictions = []
    for index, transform in enumerate(transforms):
        u = transform(x)
        predictions.append([diagnostic(u) for diagnostic in diagnostics[index]])
    return predictions