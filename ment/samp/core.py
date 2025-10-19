from typing import Callable

import torch

from ..utils import random_shuffle
from ..utils import random_uniform


class Sampler:
    """Base class for particle samplers."""

    def __init__(
        self,
        ndim: int,
        verbose: int = 0,
        device: torch.device = None,
        seed: int = None,
        noise: float = 0.0,
        noise_type: float = "gaussian",
        shuffle: bool = False,
    ) -> None:
        self.ndim = ndim
        self.verbose = verbose
        self.device = device

        self.seed = seed
        self.rng = torch.Generator()
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

        self.noise = noise
        self.noise_type = noise_type
        self.shuffle = shuffle

        self.results = {}

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        x_add = torch.zeros(x.shape, device=self.device)
        if self.noise_type == "uniform":
            x_add = random_uniform(-0.5, 0.5, device=device, rng=self.rng)
            x_add = x_add * self.noise_scale
        elif self.noise_type == "gaussian":
            x_add = torch.randn(x.shape)
            x_add = x_add * self.noise_scale
        return x + x_add

    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, prob_func: Callable, size: int) -> torch.Tensor:
        self.results = {}
        x = self._sample(prob_func, size)
        if self.noise:
            x = self.add_noise(x)
        if self.shuffle:
            x = random_shuffle(x)
        return x
