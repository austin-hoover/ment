from collections.abc import Callable
from collections.abc import Sequence

import torch

from ..diag import Histogram
from ..utils import unravel


class Transform:
    def __init__(self, device: torch.device | str = None) -> None:
        self.device = torch.device(device) if device is not None else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def to(self, device: torch.device | str) -> "Transform":
        """Move tensor state to ``device`` in place."""
        self.device = torch.device(device)
        return self

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
        super().__init__(self.matrix.device)

    def to(self, device: torch.device | str) -> "LinearTransform":
        super().to(device)
        self.matrix = self.matrix.to(self.device)
        self.matrix_inv = self.matrix_inv.to(self.device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.T

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix_inv.T


class ProjectionTransform1D(Transform):
    def __init__(self, direction: torch.Tensor) -> None:
        self.direction = direction
        super().__init__(self.direction.device)

    def to(self, device: torch.device | str) -> "ProjectionTransform1D":
        super().to(device)
        self.direction = self.direction.to(self.device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.direction, axis=1)[:, None]


class ComposedTransform(Transform):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = transforms

    def to(self, device: torch.device | str) -> "ComposedTransform":
        super().to(device)
        for transform in self.transforms:
            transform.to(self.device)
        return self

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
        x_out = transform(x)
        for diagnostic in diagnostics_copy[index]:
            diagnostic(x_out)
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
