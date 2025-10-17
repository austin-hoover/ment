"""No-Underrun Sampler (NURS).

https://github.com/bob-carpenter/no-underrun-sampler/tree/main
"""

from typing import Callable

import numpy as np
import tqdm
from scipy.special import logsumexp


# tree is tuple(selected[0], logp[1], left[2], right[3], logp_left[4], logp_right[5])
class Tree:
    def __init__(self) -> None:
        self.selected = None
        self.logp = None
        self.left = None
        self.right = None
        self.logp_left = None
        self.logp_right = None


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


def nurs_vectorized(
    rng: np.random.Generator,
    log_prob_func: Callable,
    theta_init: np.ndarray,
    num_draws: int,
    step_size: float,
    max_doublings: int,
    threshold: float,
    num_chains: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run vectorized NURS sampler across multiple chains.

    Args:
        rng: Random number generator.
        log_prob_func: Returns log probability density. Must accept (num_chains, dim) array.
        theta_init: Starting point(s). Shape (dim,) or (num_chains, dim).
        num_draws: Number of samples to draw per chain.
        step_size: Step size.
        max_doublings: Max number of doublings.
        threshold: Threshold for stopping condition.
        num_chains: Number of parallel chains to run.

    Returns:
        draws: Shape (num_chains, num_draws, dim)
        accepts: Shape (num_chains, num_draws)
        depths: Shape (num_chains, num_draws)
    """
    # Handle theta_init shape
    if theta_init.ndim == 1:
        dim = theta_init.size
        theta_init = np.tile(theta_init, (num_chains, 1))
    else:
        num_chains, dim = theta_init.shape

    log_step_size = np.log(step_size)
    log_threshold = np.log(threshold)

    def stopping_condition(tree):
        """Check stopping condition for a tree."""
        log_epsilon = log_threshold + log_step_size + tree["lp"]
        return (tree["left_min"] < log_epsilon) & (tree["right_min"] < log_epsilon)

    def leaf(theta: np.ndarray):
        """Create leaf node for a single theta."""
        lp = log_prob_func(theta.reshape(1, -1))[0]
        return {
            "selected": theta.copy(),
            "lp": lp,
            "left": theta.copy(),
            "right": theta.copy(),
            "left_min": lp,
            "right_min": lp,
        }

    def combine_trees(tree1, tree2, direction):
        """Combine two trees for a single chain."""
        lp1 = tree1["lp"]
        lp2 = tree2["lp"]
        lp12 = logsumexp([lp1, lp2])

        update = rng.binomial(1, np.exp(lp2 - lp12))
        selected = tree2["selected"] if update else tree1["selected"]

        if direction == 1:
            return {
                "selected": selected,
                "lp": lp12,
                "left": tree1["left"],
                "right": tree2["right"],
                "left_min": tree1["left_min"],
                "right_min": tree2["right_min"],
            }
        else:
            return {
                "selected": selected,
                "lp": lp12,
                "left": tree2["left"],
                "right": tree1["right"],
                "left_min": tree2["left_min"],
                "right_min": tree1["right_min"],
            }

    def build_tree(depth, theta_last, rho, direction):
        """Build tree for a single chain. Returns None if stopping condition met."""
        h = step_size * (2 * direction - 1)

        if depth == 0:
            theta_next = theta_last + h * rho
            return leaf(theta_next)

        tree1 = build_tree(depth - 1, theta_last, rho, direction)
        if tree1 is None:
            return None

        theta_mid = tree1["right"] if direction == 1 else tree1["left"]
        tree2 = build_tree(depth - 1, theta_mid, rho, direction)
        if tree2 is None:
            return None

        tree = combine_trees(tree1, tree2, direction)

        if not stopping_condition(tree):
            return None

        return tree

    def random_direction():
        """Generate random unit direction."""
        u = rng.normal(size=dim)
        return u / np.linalg.norm(u)

    def metropolis(theta, rho):
        """Metropolis step for a single chain."""
        lp_theta = log_prob_func(theta.reshape(1, -1))[0]
        s = (rng.random() - 0.5) * step_size
        theta_star = theta + s * rho
        lp_theta_star = log_prob_func(theta_star.reshape(1, -1))[0]

        accept_prob = np.minimum(1.0, np.exp(lp_theta_star - lp_theta))
        accept = rng.binomial(1, accept_prob)

        return (theta_star if accept else theta), accept

    def transition_single(theta):
        """Single transition for one chain."""
        rho = random_direction()
        theta, accept = metropolis(theta, rho)
        tree = leaf(theta)

        directions = rng.integers(0, 2, size=max_doublings)
        tree_depth = 0

        for d in range(max_doublings):
            direction = directions[d]
            theta_mid = tree["right"] if direction == 1 else tree["left"]
            tree_next = build_tree(d, theta_mid, rho, direction)

            if tree_next is None:
                break

            tree = combine_trees(tree, tree_next, direction)
            tree_depth = d

            if not stopping_condition(tree):
                break

        return tree["selected"], accept, tree_depth

    # Main sampling loop
    draws = np.zeros((num_chains, num_draws, dim))
    accepts = np.zeros((num_chains, num_draws), dtype=int)
    depths = np.zeros((num_chains, num_draws), dtype=int)

    draws[:, 0, :] = theta_init

    for m in tqdm.tqdm(range(1, num_draws), initial=1, total=num_draws):
        for chain_idx in range(num_chains):
            result = transition_single(draws[chain_idx, m - 1, :])
            draws[chain_idx, m, :] = result[0]
            accepts[chain_idx, m] = result[1]
            depths[chain_idx, m] = result[2]

    return draws, accepts, depths
