from typing import Callable

import torch

from .core import Sampler
from ..utils import edges_to_coords
from ..utils import get_grid_points
from ..utils import random_shuffle
from ..utils import random_uniform


class GridSampler(Sampler):
    """Samples from distribution on regular grid."""

    def __init__(
        self,
        limits: list[tuple[float]],
        shape: tuple[int],
        noise: float = 0.0,
        store: bool = True,
        **kws,
    ) -> None:
        super().__init__(self, **kws)

        self.shape = shape
        self.limits = limits
        self.ndim = len(limits)
        self.noise = noise
        self.store = store

        self.edges = [
            torch.linspace(
                self.limits[axis][0],
                self.limits[axis][1],
                self.shape[axis] + 1,
            )
            for axis in range(self.ndim)
        ]
        self.coords = [edges_to_coords(e) for e in self.edges]
        self.points = None

    def get_grid_points(self) -> torch.Tensor:
        if self.points is not None:
            return self.points

        points = get_grid_points(self.coords)

        if self.store:
            self.points = points
        return points

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def sample_values(self, values: torch.Tensor, size: int) -> torch.Tensor:
        """Sample from values defined on this sampler's flattened grid."""
        values_sum = torch.sum(values)
        if values_sum <= 0.0:
            raise RuntimeError("Probability is zero on the grid sampler domain.")

        idx = torch.multinomial(
            values / values_sum,
            num_samples=int(size),
            replacement=True,
            generator=self.rng,
        )
        unraveled = torch.unravel_index(idx, self.shape)

        x = torch.zeros((int(size), self.ndim), device=self.device)
        for axis in range(self.ndim):
            lb = self.edges[axis][unraveled[axis]].to(device=x.device)
            ub = self.edges[axis][unraveled[axis] + 1].to(device=x.device)
            x[:, axis] = random_uniform(
                lb,
                ub,
                int(size),
                device=self.device,
                rng=self.rng,
            )

            if self.noise:
                delta = (ub - lb) * self.noise
                x[:, axis] += 0.5 * random_uniform(
                    -delta,
                    delta,
                    int(size),
                    device=self.device,
                    rng=self.rng,
                )

        if self.shuffle:
            x = random_shuffle(x, rng=self.rng)
        return torch.squeeze(x)

    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        values = prob_func(self.get_grid_points())
        return self.sample_values(values, size)
