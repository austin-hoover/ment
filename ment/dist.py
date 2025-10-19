from typing import Callable

import numpy as np
import torch
import scipy.stats


def corrupt(x, scale, rng=None):
    return x + rng.normal(scale=scale, size=x.shape)


def decorrelate(x, rng=None):
    if x.shape[1] % 2 == 0:
        for i in range(0, d, 2):
            j = 2 * i
            idx = rng.permutation(np.arange(n))
            x[:, j : j + 1] = x[idx, j : j + 1]
    else:
        for i in range(0, d, 1):
            idx = rng.permutation(np.arange(n))
            x[:, j] = x[idx, j]
    return x


def normalize(x):
    x = x - np.mean(x, axis=0)
    x = x / np.std(x, axis=0)
    return x


def shuffle(x, rng=None):
    return rng.permutation(x)


class Distribution:
    def __init__(
        self,
        ndim: int = 2,
        seed: int = None,
        normalize: bool = False,
        shuffle: bool = True,
        noise: bool = None,
        decorr: bool = False,
        transform: Callable = None,
    ):
        self.ndim = ndim
        self.rng = np.random.default_rng(seed)
        self.normalize = normalize
        self.noise = noise
        self.shuffle = shuffle
        self.decorr = decorr
        self.transform = transform

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, size: int) -> torch.Tensor:
        x = self._sample(int(size))
        if self.shuffle:
            x = shuffle(x, rng=self.rng)
        if self.normalize:
            x = normalize(x)
        if self.noise:
            x = corrupt(x, self.noise, rng=self.rng)
        if self.decorr:
            x = decorrelate(x, rng=self.rng)
        if self.transform is not None:
            x = self.transform(x)
        x = torch.from_numpy(x)
        return x

    def _sample(self, n: int) -> np.ndarray:
        raise NotImplementedError


class EightGaussians(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.20

    def _sample(self, n):
        theta = 2.0 * np.pi * self.rng.integers(0, 8, n) / 8.0
        x = np.cos(theta)
        y = np.sin(theta)
        X = np.stack([x, y], axis=-1)
        X *= 1.5
        return X


class Galaxy(Distribution):
    def __init__(self, turns=5, truncate=3.0, **kws):
        super().__init__(**kws)
        self.turns = turns
        self.truncate = truncate
        if self.noise is None:
            self.noise = 0.0

    def _sample(self, n):
        def _rotate(X, theta):
            x = X[:, 0].copy()
            y = X[:, 1].copy()
            X[:, 0] = x * np.cos(theta) + y * np.sin(theta)
            X[:, 1] = y * np.cos(theta) - x * np.sin(theta)
            return X

        # Start with flattened Gaussian distribution.
        X = np.zeros((n, 2))
        X[:, 0] = 1.0 * scipy.stats.truncnorm.rvs(-self.truncate, self.truncate, size=n)
        X[:, 1] = 0.5 * scipy.stats.truncnorm.rvs(-self.truncate, self.truncate, size=n)

        # Apply amplitude-dependent phase advance.
        r = np.linalg.norm(X, axis=1)
        r = r / np.max(r)
        theta = 2.0 * np.pi * (1.0 + 0.5 * (r**0.25))
        for _ in range(self.turns):
            X = _rotate(X, theta)

        # Standardize the data set.
        X = X / np.std(X, axis=0)
        X = X * 0.85
        return X


class Gaussian(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)

    def _sample(self, n):
        X = self.rng.normal(size=(n, 2))
        return X


class GaussianMixture(Distribution):
    def __init__(self, modes=7, xmax=3.0, scale=0.75, shiftscale=True, **kws):
        super().__init__(**kws)
        self.modes = modes
        self.locs = self.rng.uniform(-xmax, xmax, size=(self.modes, self.ndim))
        self.scales = scale * np.ones(self.modes)
        self.shiftscale = shiftscale

    def _sample(self, n: int) -> np.ndarray:
        x = [
            self.rng.normal(loc=loc, scale=scale, size=(n // self.modes, self.ndim))
            for scale, loc in zip(self.scales, self.locs)
        ]
        x = np.vstack(x)
        if self.shiftscale:
            x = x - np.mean(x, axis=0)
            x = x / np.std(x, axis=0)
        return x


class Hollow(Distribution):
    def __init__(self, exp=1.66, **kws):
        super().__init__(**kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        x = self.rng.normal(size=(n, self.ndim))
        x /= np.linalg.norm(x, axis=1)[:, None]
        x /= np.std(x, axis=0)

        r = self.rng.uniform(0.0, 1.0, size=n) ** (1.0 / (self.exp * self.ndim))
        x *= r[:, None]
        x /= np.std(x, axis=0)
        return x


class Pinwheel(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.10

    def _sample(self, n):
        theta = 2.0 * np.pi * self.rng.integers(0, 5, n) / 5.0
        a = self.rng.normal(loc=1.0, scale=0.25, size=n)
        b = self.rng.normal(scale=0.1, size=n)
        theta = theta + np.exp(a - 1.0)
        x = a * np.cos(theta) - b * np.sin(theta)
        y = a * np.sin(theta) + b * np.cos(theta)
        X = np.stack([x, y], axis=-1)
        X = X / np.std(X, axis=0)
        return X


class Rings(Distribution):
    def __init__(self, n_rings=2, **kws):
        super().__init__(**kws)
        self.n_rings = n_rings
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, n):
        n_outer = n // self.n_rings
        sizes = [n - (self.n_rings - 1) * n_outer] + (self.n_rings - 1) * [n_outer]
        radii = np.linspace(0.0, 1.0, self.n_rings + 1)
        radii = radii[1:]

        x = []
        for size, radius in zip(sizes, radii):
            x_loc = self.rng.normal(size=(n, self.ndim))
            x_loc /= np.linalg.norm(x_loc, axis=1)[:, None]
            x_loc /= np.std(x_loc, axis=0)
            x.append(radius * x_loc)
        x = np.vstack(x)
        x /= np.std(x, axis=0)
        return x


class SwissRoll(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, n):
        t = 1.5 * np.pi * (1.0 + 2.0 * self.rng.uniform(0.0, 1.0, size=n))
        x = np.stack([t * np.cos(t), t * np.sin(t)], axis=-1)
        x = x / np.std(x, axis=0)
        return x


class TwoSpirals(Distribution):
    def __init__(self, exp=0.65, **kws):
        super().__init__(**kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.070

    def _sample(self, n):
        self.exp = 0.75
        t = 3.0 * np.pi * np.random.uniform(0.0, 1.0, size=n) ** self.exp
        r = t / 2.0 / np.pi * np.sign(self.rng.normal(size=n))
        t = t + self.rng.normal(size=n, scale=np.linspace(0.0, 1.0, n))
        x = np.stack([-r * np.cos(t), r * np.sin(t)], axis=-1)
        x = x / np.std(x, axis=0)
        return x


class KV(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        x = self.rng.normal(size=(n, self.ndim))
        x /= np.linalg.norm(x, axis=1)[:, None]
        x /= np.std(x, axis=0)
        return x


class WaterBag(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)

    def _sample(self, n):
        x = self.rng.normal(size=(n, self.ndim))
        x /= np.linalg.norm(x, axis=1)[:, None]
        x /= np.std(x, axis=0)

        r = self.rng.uniform(0.0, 1.0, size=x.shape[0]) ** (1.0 / self.ndim)
        x *= r[:, None]
        x /= np.std(x, axis=0)
        return x


DISTRIBUTIONS = {
    "eight-gaussians": EightGaussians,
    "galaxy": Galaxy,
    "gaussian-mixture": GaussianMixture,
    "hollow": Hollow,
    "kv": KV,
    "pinwheel": Pinwheel,
    "rings": Rings,
    "swissroll": SwissRoll,
    "two-spirals": TwoSpirals,
    "waterbag": WaterBag,
}


def get_dist(name, **kws):
    return DISTRIBUTIONS[name](**kws)
