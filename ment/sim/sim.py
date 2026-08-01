from collections.abc import Callable
from collections.abc import Sequence

import torch

from ..diag import Histogram
from ..utils import unravel


class Transform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityTransform(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearTransform(Transform):
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = torch.as_tensor(matrix)
        self.matrix_inv = torch.linalg.inv(self.matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=x.device, dtype=x.dtype)
        return torch.matmul(x, matrix.T)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        matrix_inv = self.matrix_inv.to(device=x.device, dtype=x.dtype)
        return torch.matmul(x, matrix_inv.T)


class ComposedTransform(Transform):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        for transform in self.transforms:
            u = transform(u)
        return u

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        x = u
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return x


def copy_histograms(histograms: Sequence[Sequence[Histogram]]) -> list[list[Histogram]]:
    return [[h.copy() for h in group] for group in histograms]


def simulate(
    x: torch.Tensor,
    transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    diagnostics: Sequence[Sequence[Histogram]],
) -> list[list[Histogram]]:
    diagnostics_copy = copy_histograms(diagnostics)
    for index, transform in enumerate(transforms):
        u = transform(x)
        for diagnostic in diagnostics_copy[index]:
            diagnostic(u)
    return diagnostics_copy


def simulate_with_diag_update(
    x: torch.Tensor,
    transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    diagnostics: Sequence[Sequence[Histogram]],
    **diag_kws,
) -> list[list[Histogram]]:

    diagnostics_copy = copy_histograms(diagnostics)
    for diagnostic in unravel(diagnostics_copy):
        for key, val in diag_kws.items():
            setattr(diagnostic, key, val)
    return simulate(x, transforms, diagnostics_copy)
