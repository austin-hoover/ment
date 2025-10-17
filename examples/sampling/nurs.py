"""No-Underrun Sampler (NURS).

https://github.com/bob-carpenter/no-underrun-sampler/tree/main
"""

from typing import Callable

import numpy as np
import tqdm
from scipy.special import logsumexp


# tree is tuple(selected[0], logp[1], left[2], right[3], logp_left[4], logp_right[5])

def nurs(
    rng: np.random.Generator,
    log_prob_func: Callable,
    theta_init: np.ndarray,
    num_draws: int,
    step_size: float,
    max_doublings: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run base NURS sampler.
    
    Args:
        rng: Random number generator.
        log_prob_func: Returns log probability density at theta.
        theta_init: Starting point.
        num_draws: Number of samples to draw.
        step_size: Step size.
        max_doublings: Max number of doublings.
        threshold: Threshold for [...]
    
    Returns:
        draws:
        accepts:
        depths: 
    """
    dim = np.size(theta_init)
    log_step_size = np.log(step_size)
    log_threshold = np.log(threshold)

    def stopping_condition(tree):
        log_epsilon = log_threshold + log_step_size + tree[1]
        return tree[4] < log_epsilon and tree[5] < log_epsilon

    def leaf(theta: np.ndarray) -> tuple:
        lp = log_prob_func(theta)
        return (theta, lp, theta, theta, lp, lp)

    def combine_trees(tree1, tree2, direction):
        lp1 = tree1[1]
        lp2 = tree2[1]
        lp12 = logsumexp([lp1, lp2])
        update = rng.binomial(1, np.exp(lp2 - lp12))
        selected = tree2[0] if update else tree1[0]
        if direction == 1:
            return (selected, lp12, tree1[2], tree2[3], tree1[4], tree2[5])
        else:
            return (selected, lp12, tree2[2], tree1[3], tree2[4], tree1[5])

    # return None if stopping condition met
    def build_tree(depth, theta_last, rho, direction):
        h = step_size * (2 * direction - 1)
        if depth == 0:
            theta_next = theta_last + h * rho
            return leaf(theta_next)
        tree1 = build_tree(depth - 1, theta_last, rho, direction)
        if not tree1:
            return None
        theta_mid = tree1[3] if direction == 1 else tree1[2]
        tree2 = build_tree(depth - 1, theta_mid, rho, direction)
        if not tree2:
            return None
        tree = combine_trees(tree1, tree2, direction)
        if stopping_condition(tree):
            return None
        return tree

    def random_direction():
        u = rng.normal(size=dim)
        return u / np.linalg.norm(u)

    def metropolis(theta, rho):
        lp_theta = log_prob_func(theta)  # computed twice (also by leaf)
        s = (rng.random() - 0.5) * step_size
        theta_star = theta + s * rho
        lp_theta_star = log_prob_func(theta_star)
        accept_prob = np.min([1.0, np.exp(lp_theta_star - lp_theta)])
        accept = rng.binomial(1, accept_prob)
        return (theta_star if accept else theta), accept

    def transition(theta):
        rho = random_direction()
        theta, accept = metropolis(theta, rho)
        tree = leaf(theta)
        directions = rng.integers(0, 2, size=max_doublings)
        for tree_depth in range(max_doublings):
            direction = directions[tree_depth]
            theta_mid = tree[3] if direction == 1 else tree[2]
            tree_next = build_tree(tree_depth, theta_mid, rho, direction)
            if not tree_next:
                break
            tree = combine_trees(tree, tree_next, direction)
            if stopping_condition(tree):
                break
        return tree[0], accept, tree_depth

    def sample():
        draws = np.zeros((num_draws, dim))
        accepts = np.zeros(num_draws, int)
        depths = np.zeros(num_draws, int)
        draws[0] = theta_init
        for m in tqdm.tqdm(range(1, num_draws), initial=1, total=num_draws):
            draws[m, :], accepts[m], depths[m] = transition(draws[m - 1])
        return draws, accepts, depths

    return sample()
