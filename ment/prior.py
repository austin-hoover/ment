import numpy as np


class GaussianPrior:
    def __init__(self, ndim: int, scale: np.ndarray) -> None:
        self.ndim = ndim
        self.scale = scale
    
    def prob(self, x: np.ndarray) -> np.ndarray:
        denom = np.sqrt((2.0 * np.pi) ** self.ndim) * np.sqrt(np.prod(self.scale))
        _prob = np.exp(-0.5 * np.sum(np.square(x / self.scale), axis=1))
        _prob = _prob / denom
        return _prob
        

class UniformPrior:
    def __init__(self, ndim: int = 2, scale: float = 10.0) -> None:
        self.scale = scale
        self.ndim = ndim
        self.volume = scale ** ndim

    def prob(self, x: np.ndarray) -> np.ndarray:
        _prob = np.ones(x.shape[0]) / self.volume
        for j in range(x.shape[1]):
            _in_bounds = np.abs(x[:, j]) <= 0.5 * self.scale
            _prob *= _in_bounds.astype(float)
        return _prob