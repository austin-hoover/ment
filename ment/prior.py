from typing import Callable
import numpy as np
import scipy.special


def sphere_volume(ndim: int, radius: float) -> float:
    factor = (np.pi ** (0.5 * ndim)) / scipy.special.gamma(1.0 + 0.5 * ndim)
    return factor * (radius ** ndim)


class Prior:
    def __init__(self, ndim: int, **kws) -> None:
        self.ndim = ndim

    def prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, size: int) -> np.ndarray:
        raise NotImplementedError


class GaussianPrior(Prior):
    def __init__(self, scale: np.ndarray, **kws) -> None:
        super().__init__(**kws)
        self.scale = scale

    def prob(self, x: np.ndarray) -> np.ndarray:
        denom = np.sqrt((2.0 * np.pi) ** self.ndim) * np.sqrt(np.prod(self.scale))
        prob = np.exp(-0.5 * np.sum(np.square(x / self.scale), axis=1))
        prob = prob / denom
        return prob

    def sample(self, size: int) -> np.ndarray:
        x = np.random.normal(size=(size, self.ndim))
        x = x * self.scale
        return x
    
        
class UniformPrior(Prior):
    def __init__(self, scale: float = 10.0, **kws) -> None:
        super().__init__(**kws)
        self.scale = scale
        self.volume = scale**self.ndim

    def prob(self, x: np.ndarray) -> np.ndarray:
        prob = np.ones(x.shape[0]) / self.volume
        for j in range(x.shape[1]):
            in_bounds = np.abs(x[:, j]) <= 0.5 * self.scale
            prob *= in_bounds.astype(float)
        return prob

    def sample(self, size: int) -> np.ndarray:
        x = np.random.uniform(-0.5, 0.5, size=(size, self.ndim))
        x = x * self.scale
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
