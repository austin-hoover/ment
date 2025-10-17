"""Common prior distributions.

Note that MENT priors do not need to be normalized!
"""
from typing import Callable
import numpy as np
import scipy.special
import scipy.stats
import warnings

from .samp import MetropolisHastingsSampler


def sphere_volume(ndim: int, radius: float) -> float:
    factor = (np.pi ** (0.5 * ndim)) / scipy.special.gamma(1.0 + 0.5 * ndim)
    return factor * (radius**ndim)


class Prior:
    def __init__(self, ndim: int, **kws) -> None:
        self.ndim = ndim

    def prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, size: int) -> np.ndarray:
        raise NotImplementedError


class GaussianPrior(Prior):
    def __init__(self, scale: np.ndarray, trunc: float = None, **kws) -> None:
        super().__init__(**kws)
        self.scale = scale
        self.trunc = trunc

    def prob(self, x: np.ndarray) -> np.ndarray:
        denom = np.sqrt((2.0 * np.pi) ** self.ndim) * np.sqrt(np.prod(self.scale))
        prob = np.exp(-0.5 * np.sum(np.square(x / self.scale), axis=1))
        prob = prob / denom

        # This will give the **unnormalized** density
        if self.trunc is not None:
            r = np.linalg.norm(x, axis=1)
            prob[r > self.trunc] = 0.0

        return prob

    def sample(self, size: int) -> np.ndarray:
        if self.trunc is not None:
            warnings.warn("Using MCMC to generate approximate samples.")

            chains = 1
            start = np.random.uniform(-self.scale, self.scale, size=(chains, self.ndim))
            proposal_cov = np.eye(self.ndim) * self.scale**2 * 1.0

            sampler = MetropolisHastingsSampler(
                ndim=self.ndim,
                chains=chains,
                burnin=1000,
                start=start,
                proposal_cov=proposal_cov,
                shuffle=True,
                verbose=0,
            )
            x = sampler.sample(self.prob, size)
            return x

        x = np.random.normal(size=(size, self.ndim))
        x = x * self.scale
        return x


class RectangularTruncatedGaussianPrior(Prior):
    """Product of marginal truncated Gaussian priors (square boundary)."""

    def __init__(self, scale: np.ndarray, trunc: np.ndarray, **kws) -> None:
        super().__init__(**kws)

        self.scale = scale
        if np.ndim(self.scale) == 0:
            self.scale = self.scale * np.ones(self.ndim)

        self.trunc = trunc
        if np.ndim(self.trunc) == 0:
            self.trunc = self.trunc * np.ones(self.ndim)

        self.marg_dists = []
        for i in range(self.ndim):
            marg_dist = scipy.stats.truncnorm(
                -self.trunc[i], +self.trunc[i], scale=self.scale[i]
            )
            self.marg_dists.append(marg_dist)

    def prob(self, x: np.ndarray) -> np.ndarray:
        prob = np.ones(x.shape[0])
        for j, marg_dist in enumerate(self.marg_dists):
            prob *= marg_dist.pdf(x[:, j])
        return prob

    def sample(self, size: int) -> np.ndarray:
        x = np.zeros((size, self.ndim))
        for j, marg_dist in enumerate(self.marg_dists):
            x[:, j] = marg_dist.rvs(size)
        return x


class UniformPrior(Prior):
    def __init__(self, scale: float = 10.0, **kws) -> None:
        super().__init__(**kws)
        self.scale = scale
        self.volume = scale**self.ndim

    def prob(self, x: np.ndarray) -> np.ndarray:
        prob = np.ones(x.shape[0]) / self.volume
        for j in range(x.shape[1]):
            in_bounds = np.abs(x[:, j]) <= self.scale
            prob *= in_bounds.astype(float)
        return prob

    def sample(self, size: int) -> np.ndarray:
        x = np.random.uniform(-self.scale, self.scale, size=(size, self.ndim))
        return x


class UniformSphericalPrior(Prior):
    def __init__(self, scale: float = 10.0, **kws) -> None:
        super().__init__(**kws)
        self.scale = scale
        self.volume = sphere_volume(ndim=self.ndim, radius=self.scale)

    def prob(self, x: np.ndarray) -> np.ndarray:
        prob = np.ones(x.shape[0]) / self.volume
        r = np.sqrt(np.sum(np.square(x), axis=1))
        prob[r > self.scale] = 0.0
        return prob

    def sample(self, size: int) -> np.ndarray:
        # Sample from normal distribution
        x = np.random.normal(size=(size, self.ndim))

        # Scale to unit norm (KV distribution)
        scale = 1.0 / np.sqrt(np.sum(x**2, axis=1))
        x = x * scale[:, None]

        # Fill sphere with uniform density
        scale = np.random.uniform(0.0, 1.0, size=size) ** (1.0 / self.ndim)
        scale = scale * self.scale
        x = x * scale[:, None]
        return x
