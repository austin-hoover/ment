import numpy as np
import ment


def make_dist(size: int, seed: int = None) -> np.ndarray:
    """Make ground truth 4D phase space distribution."""
    rng = np.random.default_rng(seed)
    x = ment.dist.Rings(ndim=4, noise=0.10, seed=seed).sample(size)
    x = x / np.std(x, axis=0)
    return x


def get_actions(x: np.ndarray) -> np.ndarray:
    """Return actions J1 and J2 from 4D phase space coordinate array."""
    cov_matrix = np.cov(x.T)
    norm_matrix = ment.cov.normalization_matrix(cov_matrix, scale=False)
    z = np.matmul(x, norm_matrix.T)

    actions = []
    for i in range(0, z.shape[1], 2):
        axis = (i, i + 1)
        action = np.sum(np.square(z[:, axis]), axis=1)
        actions.append(action)
        
    actions = np.stack(actions, axis=-1)
    return actions
