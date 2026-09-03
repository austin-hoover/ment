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
        self.device = torch.device(device) if device is not None else None

        self.seed = seed
        # PyTorch supports explicitly constructed generators for CPU and CUDA,
        # but not MPS. The global generator is correctly selected for MPS ops.
        self.rng = None
        if self.device is None or self.device.type == "cpu":
            self.rng = torch.Generator()
        if self.seed is not None:
            if self.rng is None:
                torch.manual_seed(self.seed)
            else:
                self.rng.manual_seed(self.seed)

        self.noise = noise
        self.noise_type = noise_type
        self.shuffle = shuffle

        self.results = {}

    def to(self, device: torch.device | str) -> "Sampler":
        """Configure sampling and random-number generation for ``device``."""
        self.device = torch.device(device)
        if self.device.type == "cpu":
            self.rng = torch.Generator()
            if self.seed is not None:
                self.rng.manual_seed(self.seed)
        else:
            self.rng = None
            if self.seed is not None:
                torch.manual_seed(self.seed)
        return self

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        x_add = torch.zeros_like(x)
        if self.noise_type == "uniform":
            x_add = random_uniform(-0.5, 0.5, device=self.device, rng=self.rng)
            x_add = x_add * self.noise
        elif self.noise_type == "gaussian":
            x_add = torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=self.rng
            )
            x_add = x_add * self.noise
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
