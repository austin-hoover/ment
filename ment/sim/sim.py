from copy import deepcopy
from typing import Callable

import numpy as np

from ..utils import unravel


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


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def forward(
    x: np.ndarray,
    transforms: list[Callable],
    diagnostics: list[list[Callable]],
) -> list[list[np.ndarray]]:
    projections = []
    for index, transform in enumerate(transforms):
        u = transform(x)
        projections.append([diagnostic(u) for diagnostic in diagnostics[index]])
    return projections


def forward_with_diag_update(
    x: np.ndarray,
    transforms: list[Callable],
    diagnostics: list[list[Callable]],
    **diagnostic_kws,
) -> list[list[np.ndarray]]:

    diagnostics_new = deepcopy(diagnostics)
    for diagnostic in unravel(diagnostics_new):
        for key, val in diagnostic_kws.items():
            setattr(diagnostic, key, val)
    return forward(x, transforms, diagnostics_new)
