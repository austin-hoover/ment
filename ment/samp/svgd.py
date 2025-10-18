from typing import Callable

import numpy as np
import torch

from .core import Sampler
from ..utils import wrap_tqdm


class RBFKernel(torch.nn.Module):
    def __init__(self, sigma: torch.Tensor = None) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xx = torch.matmul(x, x.T)
        xy = torch.matmul(x, y.T)
        yy = torch.matmul(y, y.T)
        
        dnorm2 = -2.0 * xy + xx.diag().unsqueeze(1) + yy.diag().unsqueeze(0)

        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(x.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1.00e-08 + 2.0 * sigma**2)
        k_xy = torch.exp(-gamma * dnorm2)
        return k_xy


class SVGDSampler(Sampler):
    def __init__(self, kernel: torch.nn.Module, train_kws: dict = None, verbose: int = 0, sample_prior: Callable = None, **kws) -> None:
        super().__init__(**kws)
        self.kernel = kernel
        self.verbose = verbose
        self.results = {}

        self.sample_prior = sample_prior
        if self.sample_prior is None:
            self.sample_prior = lambda size: torch.randn(size, self.ndim)
        
        self.train_kws = train_kws
        if self.train_kws is None:
            self.train_kws = dict()
            
        self.train_kws.setdefault("iters", 100)
        self.train_kws.setdefault("lr", 0.1)

    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        self.results = {}
        self.results["history"] = []
        
        x = self.sample_prior(size)
        x.requires_grad_(True)
        self.results["history"].append(x.clone().detach())

        optimizer = torch.optim.Adam([x], lr=self.train_kws["lr"])
        
        for _ in wrap_tqdm(range(self.train_kws["iters"]), self.verbose):
            optimizer.zero_grad()

            log_prob = torch.log(prob_func(x) + 1.00e-12)
            score = torch.autograd.grad(torch.sum(log_prob), x)[0]
            
            k_xx = self.kernel(x, x.detach())
            grad_k = -torch.autograd.grad(torch.sum(k_xx), x)[0]
            phi = (k_xx.detach().matmul(score) + grad_k) / x.size(0)

            x.grad = -phi
            optimizer.step()

            self.results["history"].append(x.clone().detach())

        return x.detach()
