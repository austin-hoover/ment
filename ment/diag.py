import copy
from typing import Self

import numpy as np
import scipy.ndimage
import torch

from .utils import coords_to_edges
from .utils import edges_to_coords
from .utils import get_grid_points
from .utils import weighted_average


class Histogram:
    def __init__(
        self, blur: float = 0.0, thresh: float = 0.0, thresh_type: str = "abs"
    ) -> None:
        self.blur = blur
        self.thresh = thresh
        self.thresh_type = thresh_type

    def to(self, device: torch.device | str) -> "Histogram":
        """Move the histogram's tensor state to ``device`` in place."""
        device = torch.device(device)
        for name in (
            "values",
            "coords",
            "edges",
            "bin_sizes",
            "bin_size",
            "bin_volume",
            "bin_width",
            "direction",
        ):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
            elif isinstance(value, list):
                setattr(
                    self,
                    name,
                    [
                        item.to(device) if isinstance(item, torch.Tensor) else item
                        for item in value
                    ],
                )
        return self


class HistogramND(Histogram):
    def __init__(
        self,
        axis: tuple[int, ...],
        edges: list[torch.Tensor] = None,
        coords: list[torch.Tensor] = None,
        values: torch.Tensor = None,
        **kws,
    ) -> None:
        super().__init__(**kws)

        self.axis = axis
        self.ndim = len(axis)

        self.values = values
        if self.values is not None:
            self.values = self.values.float()

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = [edges_to_coords(e) for e in self.edges]
        if self.edges is None and self.coords is not None:
            self.edges = [coords_to_edges(c) for c in self.coords]
        for i in range(len(self.coords)):
            self.coords[i] = self.coords[i].float()
        for i in range(len(self.edges)):
            self.edges[i] = self.edges[i].float()

        self.bin_sizes = [
            self.coords[i][1] - self.coords[i][0] for i in range(self.ndim)
        ]
        self.bin_sizes = torch.stack(self.bin_sizes)
        self.bin_volume = torch.prod(self.bin_sizes)
        self.grid_shape = tuple([len(c) for c in self.coords])

        self.shape = tuple([len(c) for c in self.coords])
        self.values = values
        if self.values is None:
            self.values = torch.zeros(self.shape, device=self.coords[0].device)
        self.normalize()

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def normalize(self) -> None:
        values = torch.clone(self.values)
        values_sum = torch.sum(values)
        if values_sum > 0.0:
            values = values / values_sum / self.bin_volume
        self.values = values

    def process(self) -> None:
        values = torch.clone(self.values)

        if self.blur:
            device = values.device
            values = values.cpu().numpy()
            values = scipy.ndimage.gaussian_filter(values, self.blur)
            values = torch.from_numpy(values)
            values = values.to(device)

        if self.thresh > 0:
            min_value = self.thresh
            if self.thresh_type == "frac":
                min_value = torch.max(values) * self.thresh
            values[values < min_value] = 0.0

        self.values = values
        self.normalize()

    def get_grid_points(self) -> torch.Tensor:
        return get_grid_points(self.coords)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.axis]

    def bin(self, x: torch.Tensor) -> torch.Tensor:
        self.values = self.bin_counts(x)
        self.process()
        return self.values

    def bin_counts(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized bin counts without modifying the histogram."""
        x_proj = self.project(x)
        return torch.histogramdd(x_proj, bins=self.edges, density=False).hist

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.bin(x)

    def cov(self) -> torch.Tensor:
        S = torch.zeros(
            (self.ndim, self.ndim), device=self.values.device, dtype=self.values.dtype
        )

        values_sum = torch.sum(self.values)
        if values_sum <= 0.0:
            return S

        COORDS = torch.meshgrid(*self.coords, indexing="ij")
        coords_mean = torch.stack(
            [weighted_average(C, weights=self.values) for C in COORDS]
        )
        for i in range(self.ndim):
            for j in range(i + 1):
                X = COORDS[i] - coords_mean[i]
                Y = COORDS[j] - coords_mean[j]
                EX = torch.sum(self.values * X) / values_sum
                EY = torch.sum(self.values * Y) / values_sum
                EXY = torch.sum(self.values * X * Y) / values_sum
                S[i, j] = S[j, i] = EXY - EX * EY
        return S


class Histogram1D(Histogram):
    def __init__(
        self,
        axis: int = 0,
        edges: torch.Tensor = None,
        coords: torch.Tensor = None,
        values: torch.Tensor = None,
        direction: torch.Tensor = None,
        **kws,
    ) -> None:
        super().__init__(**kws)

        self.axis = axis
        self.ndim = 1

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = edges_to_coords(self.edges)
        if self.edges is None and self.coords is not None:
            self.edges = coords_to_edges(self.coords)

        self.bin_size = self.coords[1] - self.coords[0]
        self.bin_volume = self.bin_width = self.bin_size

        self.direction = direction
        if self.direction is not None:
            self.direction /= torch.linalg.norm(self.direction)

        self.shape = len(self.coords)
        self.values = values
        if self.values is None:
            self.values = torch.zeros(self.shape, device=self.coords.device)
        self.normalize()

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def get_grid_points(self) -> torch.Tensor:
        return self.coords

    def normalize(self) -> None:
        values = torch.clone(self.values)
        values_sum = torch.sum(values)
        if values_sum > 0.0:
            values = values / values_sum / self.bin_volume
        self.values = values

    def process(self) -> None:
        values = torch.clone(self.values)

        if self.blur:
            device = values.device
            values = values.cpu().numpy()
            values = scipy.ndimage.gaussian_filter(values, self.blur)
            values = torch.from_numpy(values)
            values = values.to(device)

        if self.thresh > 0:
            min_value = self.thresh
            if self.thresh_type == "frac":
                min_value = torch.max(values) * self.thresh
            values[values < min_value] = 0.0

        self.values = values
        self.normalize()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.direction is not None:
            return torch.sum(x * self.direction, axis=1)
        return x[:, self.axis]

    def bin(self, x: torch.Tensor) -> torch.Tensor:
        self.values = self.bin_counts(x)
        self.process()
        return self.values

    def bin_counts(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized bin counts without modifying the histogram."""
        x_proj = self.project(x)
        return torch.histogram(x_proj, bins=self.edges, density=False).hist

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.bin(x)

    def var(self) -> torch.Tensor:
        values_sum = torch.sum(self.values)
        if values_sum <= 0.0:
            raise ValueError("Histogram values are zero.")

        x = self.coords
        f = self.values
        x_avg = weighted_average(x, weights=f)
        x_var = weighted_average((x - x_avg) ** 2, weights=f)
        return x_var
