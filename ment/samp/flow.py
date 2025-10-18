import time
from typing import Callable

import torch
import zuko

from .core import Sampler


class FlowSampler(Sampler):
    """Normalizing flow sampler."""
    def __init__(
        self,
        flow: zuko.flows.Flow,
        unnorm_matrix: torch.Tensor = None,
        train_kws: dict = None,
        **kws,
    ) -> None:
        super().__init__(**kws)

        self.flow = flow
        self.prob_func = None
        self.trained = False

        self.unnorm_matrix = unnorm_matrix
        if self.unnorm_matrix is None:
            self.unnorm_matrix = torch.eye(self.ndim)

        self.train_kws = train_kws
        if self.train_kws is None:
            self.train_kws = {}

        self.train_kws.setdefault("batch_size", 512)
        self.train_kws.setdefault("iters", 1000)
        self.train_kws.setdefault("lr", 0.001)
        self.train_kws.setdefault("lr_min", 0.001)
        self.train_kws.setdefault("lr_decay", 0.99)
        self.train_kws.setdefault("print_freq", 100)
        self.train_kws.setdefault("verbose", 0)

        self.results = {}
        self.results["loss"] = []
        self.results["time"] = []

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z, self.unnorm_matrix.T)

    def train(self, prob_func: Callable) -> dict:
        self.prob_func = prob_func
        self.trained = True

        self.results = {}
        self.results["loss"] = []
        self.results["time"] = []

        iters = self.train_kws["iters"]
        batch_size = self.train_kws["batch_size"]
        lr = self.train_kws["lr"]
        lr_min = self.train_kws["lr_min"]
        lr_decay = self.train_kws["lr_decay"]
        print_freq = self.train_kws["print_freq"]
        verbose = self.train_kws["verbose"]

        start_time = time.time()

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        for iteration in range(iters):
            x, log_prob = self.flow().rsample_and_log_prob((batch_size,))
            x = self.unnormalize(x)

            loss = torch.mean(log_prob) - torch.mean(torch.log(prob_func(x) + 1.00e-15))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(lr_min, lr_decay * param_group["lr"])

            # Append to history array
            self.results["loss"].append(loss.detach())
            self.results["time"].append(time.time() - start_time)

            # Print update
            if verbose and (iteration % print_freq == 0):
                print(iteration, loss)

    def __call__(self, prob_func: Callable, size: int) -> torch.Tensor:
        self.trained = False

        if not self.trained:
            self.train(prob_func)

        with torch.no_grad():
            x = self.flow().sample((size,))
            x = self.unnormalize(x)
            return x
