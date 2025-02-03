from copy import deepcopy
from typing import Callable
from typing import Union

import numpy as np

from ..diag import HistogramND
from ..diag import Histogram1D
from ..utils import unravel


type Histogram = Union[Histogram1D, HistogramND]


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


def copy_histograms(histograms: list[list[Histogram]]) -> list[list[Histogram]]:
    histograms_copy = []
    for index in range(len(histograms)):
        histograms_copy.append([histogram.copy() for histogram in histograms[index]])
    return histograms_copy


copy_diagnostics = copy_histograms
copy_projections = copy_histograms


def simulate(
    x: np.ndarray, transforms: list[Callable], diagnostics: list[list[Histogram]]
) -> list[list[Histogram]]:
    for index, transform in enumerate(transforms):
        u = transform(x)
        for diagnostic in diagnostics[index]:
            diagnostic(u)
    return copy_diagnostics(diagnostics)


def simulate_with_diag_update(
    x: np.ndarray, transforms: list[Callable], diagnostics: list[list[Histogram]], **kws
) -> list[list[Histogram]]:
    kws_list_old, kws_list_new = [], []
    for diagnostic in unravel(diagnostics):
        kws_old, kws_new = {}, {}
        for key, val in kws.items():
            kws_old[key] = getattr(diagnostic, key, val)
            kws_new[key] = val
        kws_list_old.append(kws_old)
        kws_list_new.append(kws_new)

    for diagnostic, kws in zip(unravel(diagnostics), kws_list_new):
        for key, val in kws.items():
            setattr(diagnostic, key, val)

    simulate(x, transforms, diagnostics)

    for diagnostic, kws in zip(unravel(diagnostics), kws_list_old):
        for key, val in kws.items():
            setattr(diagnostic, key, val)

    return copy_diagnostics(diagnostics)
