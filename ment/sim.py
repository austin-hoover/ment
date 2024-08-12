from typing import Callable
from typing import Union

import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


class Transform:
    def __init__(self) -> None:
        return
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        NotImplementedError

    def inverse(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x


class LinearTransform(Transform):
    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__()
        self.matrix = matrix
        self.matrix_inv = np.linalg.inv(matrix)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.matrix.T)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.matrix_inv.T)


class ComposedTransform(Transform):
    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def forward(self, x: np.ndarray) -> np.ndarray:
        u = x.copy()
        for transform in self.transforms:
            u = transform(u)
        return u

    def inverse(self, u: np.ndarray) -> np.ndarray:
        x = u.copy()
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return x


def forward(x: np.ndarray, transforms: list[Callable], diagnostics: list[list[Callable]]) -> list[list[np.ndarray]]:
    values = []
    for index, transform in enumerate(transforms):
        u = transform(x)
        values.append([diagnostic(u) for diagnostic in diagnostics[index]])
    return values


