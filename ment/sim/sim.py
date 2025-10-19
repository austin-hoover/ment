from typing import Callable

import torch

from ..diag import Histogram
from ..utils import unravel


class Transform:
    def __init__(self) -> None:
        return

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        NotImplementedError

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearTransform(Transform):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.matrix = matrix.float()
        self.matrix_inv = torch.linalg.inv(self.matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.matrix.T)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.matrix_inv.T)


class ComposedTransform(Transform):
    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        for transform in self.transforms:
            u = transform(u)
        return u

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        x = u.clone()
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return x


def copy_histograms(histograms: list[list[Histogram]]) -> list[list[Histogram]]:
    histograms_copy = []
    for index in range(len(histograms)):
        histograms_copy.append([histogram.copy() for histogram in histograms[index]])
    return histograms_copy


def simulate(
    x: torch.Tensor, transforms: list[Callable], diagnostics: list[list[Histogram]]
) -> list[list[Histogram]]:

    diagnostics_copy = copy_histograms(diagnostics)
    for index, transform in enumerate(transforms):
        u = transform(x)
        for diagnostic in diagnostics_copy[index]:
            diagnostic(u)
    return diagnostics_copy


def simulate_with_diag_update(
    x: torch.Tensor,
    transforms: list[Callable],
    diagnostics: list[list[Histogram]],
    **kws
) -> list[list[Histogram]]:
    diagnostics_copy = copy_histograms(diagnostics)
    for diagnostic in unravel(diagnostics_copy):
        for key, val in kws.items():
            setattr(diagnostic, key, val)
    return simulate(x, transforms, diagnostics_copy)
