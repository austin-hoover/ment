import numpy as np
import torch

import ment


def make_dist(size: int, seed: int = None) -> np.ndarray:
    """Make ground truth 4D phase space distribution."""
    x = ment.dist.Rings(ndim=4, noise=0.10, seed=seed).sample(size)
    x = x / torch.std(x, axis=0)
    return x


def get_actions(x: torch.Tensor) -> np.ndarray:
    """Return actions J1 and J2 from 4D phase space coordinate array."""
    cov_matrix = torch.cov(x.T)
    norm_matrix = ment.cov.build_norm_matrix_from_cov(cov_matrix, scale=False)
    z = torch.matmul(x, norm_matrix.T)

    actions = []
    for i in range(0, z.shape[1], 2):
        axis = (i, i + 1)
        action = torch.sum(torch.square(z[:, axis]), axis=1)
        actions.append(action)

    actions = torch.stack(actions, axis=-1)
    return actions
