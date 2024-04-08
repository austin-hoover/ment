import numpy as np


class GaussianPrior:
    """Gaussian prior distribution."""
    def __init__(self, dim: int, scale: np.ndarray) -> None:
        self.dim = dim
        self.scale = scale
    
    def prob(self, x):
        denom = np.sqrt((2.0 * np.pi) ** self.dim) * np.sqrt(np.prod(self.scale))
        prob = np.exp(-0.5 * np.sum(np.square(x / self.scale), axis=1))
        prob = prob / denom
        return prob
        

class UniformPrior:
    """Uniform prior distribution."""
    def __init__(self, dim: int = 2, scale: float = 10.0) -> None:
        self.scale = scale
        self.dim = dim
        self.volume = scale ** dim

    def prob(self, x):
        return (1.0 / self.volume)