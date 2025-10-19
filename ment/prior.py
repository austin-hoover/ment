"""Prior distributions.

Note that MENT priors do not need to be normalized.
"""
import math
import warnings

import numpy as np
import torch


class Prior:
    def __init__(self, ndim: int, **kws) -> None:
        self.ndim = ndim

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, size: int) -> torch.Tensor:
        raise NotImplementedError


class GaussianPrior(Prior):
    def __init__(self, scale: torch.Tensor | float, **kws) -> None:
        super().__init__(**kws)
        self.scale = torch.as_tensor(scale)
        if self.scale.numel() < self.ndim:
            self.scale = self.scale * torch.ones(self.ndim)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        denom = math.sqrt((2.0 * math.pi) ** self.ndim)
        denom *= torch.sqrt(torch.prod(self.scale))
        prob = torch.exp(-0.5 * torch.sum(torch.square(x / self.scale), axis=1))
        prob = prob / denom
        return prob


class InfiniteUniformPrior(Prior):
    def __init__(self, **kws) -> None:
        super().__init__(**kws)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0])
