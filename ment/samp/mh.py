import math
import time
from typing import Callable

import torch

from .core import Sampler
from ..utils import random_uniform
from ..utils import wrap_tqdm


class MetropolisHastingsSampler(Sampler):
    """Metropolis-Hastings sampler.

    https://colindcarroll.com/2019/08/18/very-parallel-mcmc-sampling/
    """

    def __init__(
        self,
        start: torch.Tensor,
        proposal_cov: torch.Tensor,
        burnin: int = 0,
        **kws,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        chains : int
            Number of sampling chains.
        burnin : int
            Number of burnin iterations (applies to each chain).
        start : torch.Tensor
            An array of shape (nchains, ndim) giving the starting point of each chain.
            Can also pass shape (ndim,) which will initialize a single chains.
        proposal_cov : torch.Tensor
            We use a Gaussian proposal distribution centered on the current point in
            the random walk. This variable is the covariance matrix of the Gaussian
            distribution.
        """
        super().__init__(**kws)

        self.start = start
        if self.start.ndim == 1:
            self.start = self.start[None, :]

        self.chains = self.start.shape[0]
        self.burnin = burnin

        self.proposal_mean = torch.zeros(self.ndim, device=self.device)
        self.proposal_cov = proposal_cov
        if self.proposal_cov is None:
            self.proposal_cov = torch.eye(self.ndim, device=self.device)
        self.proposal_dist = torch.distributions.MultivariateNormal(
            self.proposal_mean, self.proposal_cov
        )

    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        # Initialize list of points. From now on we each "point" is really a batch of
        # size (nchains, ndim). Burnin-points will be discarded later.
        size = int(math.ceil(size / float(self.chains)))
        size = size + self.burnin
        points = torch.zeros((size, self.chains, self.ndim), device=self.device)

        # Sample proposal points from a Gaussian distribution. (The means will be updated
        # during the random walk.)
        proposal_points = self.proposal_dist.sample((size - 1, self.chains))

        # Set starting point for each chain. If none is provided, sample from the proposal
        # distribution centered at the origin.
        start = self.start
        points[0] = start

        # Execute random walk
        random_uniforms = random_uniform(
            lb=0.0,
            ub=1.0,
            size=(size - 1, self.chains),
            rng=self.rng,
            device=self.device,
        )
        accept = torch.zeros(self.chains, device=self.device)

        points[0] = start
        prob = prob_func(start)

        t0 = time.time()

        self.results = {}
        self.results["n_total_accepted"] = 0
        self.results["n_total"] = 0
        self.results["acceptance_rate"] = None
        self.results["time"] = None
        self.results["time_per_step"] = None

        for i in wrap_tqdm(range(1, size), self.verbose):
            proposal_point = points[i - 1] + proposal_points[i - 1]
            proposal_prob = prob_func(proposal_point)
            accept = proposal_prob > prob * random_uniforms[i - 1]

            self.results["time"] = time.time() - t0
            self.results["time_per_step"] = self.results["time"] / (i + 1)

            if i > self.burnin:
                self.results["n_total_accepted"] += torch.count_nonzero(accept)
                self.results["n_total"] += self.chains
                self.results["acceptance_rate"] = (
                    self.results["n_total_accepted"] / self.results["n_total"]
                )
                if self.verbose > 2:
                    print(f"debug {i:05.0f}")
                    print(self.results)

            points[i] = points[i - 1]
            points[i][accept] = proposal_point[accept]
            prob[accept] = proposal_prob[accept]

        points = points[self.burnin :]

        # Debug
        if self.verbose > 1:
            print("debug acceptance rate =", self.results["acceptance_rate"])
            for axis in range(self.ndim):
                x_chain_stds = [torch.std(x_chain[:, axis]) for x_chain in points]
                x_chain_avgs = [torch.mean(x_chain[:, axis]) for x_chain in points]
                print(
                    f"debug axis={axis} between-chain avg(x_chain_std) =",
                    torch.mean(x_chain_stds),
                )
                print(
                    f"debug axis={axis} between-chain std(x_chain_std) =",
                    torch.std(x_chain_stds),
                )
                print(
                    f"debug axis={axis} between-chain avg(x_chain_avg) =",
                    torch.mean(x_chain_avgs),
                )
                print(
                    f"debug axis={axis} between-chain std(x_chain_avg) =",
                    torch.std(x_chain_avgs),
                )

                x_avg = torch.mean(torch.hstack([chain[:, axis] for chain in points]))
                x_std = torch.std(torch.hstack([chain[:, axis] for chain in points]))
                print(f"debug axis={axis} x_std =", x_std)
                print(f"debug axis={axis} x_avg =", x_avg)

        # Merge chains
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])

        # Make sure size is correct
        size = size * self.chains
        points = points[:size]
        return points
