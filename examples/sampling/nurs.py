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


def nurs_vec(
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

    Source: Claude Sonnet 4.5

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

    def stopping_condition(trees):
        """Check stopping condition for batch of trees."""
        log_epsilon = log_threshold + log_step_size + trees[:, 1]
        return (trees[:, 4] < log_epsilon) & (trees[:, 5] < log_epsilon)

    def leaf(theta_batch: np.ndarray) -> np.ndarray:
        """Create leaf nodes for batch of thetas.

        Returns: (num_chains, 6) where columns are:
            0: selected theta index (unused for leaf)
            1: log prob
            2-3: left/right bounds (same as theta for leaf)
            4-5: min log probs on left/right
        """
        lp = log_prob_func(theta_batch)
        n = theta_batch.shape[0]
        return np.column_stack(
            [
                np.arange(n),  # self-reference
                lp,
                np.arange(n),  # left = self
                np.arange(n),  # right = self
                lp,  # left_min
                lp,  # right_min
            ]
        )

    def combine_trees(tree1, tree2, direction, theta_storage):
        """Combine two trees. Direction: 1=forward, 0=backward."""
        lp1 = tree1[:, 1]
        lp2 = tree2[:, 1]
        lp12 = logsumexp(np.column_stack([lp1, lp2]), axis=1)

        # Randomly select from tree1 or tree2
        probs = np.exp(lp2 - lp12)
        updates = rng.binomial(1, probs).astype(bool)
        selected = np.where(updates, tree2[:, 0], tree1[:, 0]).astype(int)

        if direction == 1:
            return np.column_stack(
                [
                    selected,
                    lp12,
                    tree1[:, 2],  # left
                    tree2[:, 3],  # right
                    tree1[:, 4],  # left_min
                    tree2[:, 5],  # right_min
                ]
            )
        else:
            return np.column_stack(
                [
                    selected,
                    lp12,
                    tree2[:, 2],  # left
                    tree1[:, 3],  # right
                    tree2[:, 4],  # left_min
                    tree1[:, 5],  # right_min
                ]
            )

    def build_tree(depth, theta_storage, tree_current, rho, direction):
        """Build tree for all active chains simultaneously."""
        h = step_size * (2 * direction - 1)
        active = np.ones(num_chains, dtype=bool)

        if depth == 0:
            # Get theta from current tree's right or left boundary
            if direction == 1:
                theta_idx = tree_current[:, 3].astype(int)
            else:
                theta_idx = tree_current[:, 2].astype(int)
            theta_last = theta_storage[np.arange(num_chains), theta_idx]
            theta_next = theta_last + h * rho

            # Add to storage
            new_idx = theta_storage.shape[1]
            theta_storage = np.concatenate(
                [theta_storage, theta_next[:, :, np.newaxis]], axis=1
            )

            tree_new = leaf(theta_next)
            tree_new[:, 0] = new_idx  # Update to new index
            tree_new[:, 2] = new_idx
            tree_new[:, 3] = new_idx
            return tree_new, active, theta_storage

        # Recursive case
        tree1, active1, theta_storage = build_tree(
            depth - 1, theta_storage, tree_current, rho, direction
        )
        active &= active1

        # Update tree_current to use tree1's boundaries
        tree_mid = tree1.copy()

        tree2, active2, theta_storage = build_tree(
            depth - 1, theta_storage, tree_mid, rho, direction
        )
        active &= active2

        tree = combine_trees(tree1, tree2, direction, theta_storage)

        # Check stopping condition
        stopped = ~stopping_condition(tree)
        active &= ~stopped

        return tree, active, theta_storage

    def random_directions():
        """Generate random unit directions for all chains."""
        u = rng.normal(size=(num_chains, dim))
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        return u / norms

    def metropolis(theta_batch, rho_batch, theta_storage):
        """Vectorized Metropolis step."""
        lp_theta = log_prob_func(theta_batch)
        s = (rng.random(size=num_chains) - 0.5) * step_size
        theta_star = theta_batch + s[:, np.newaxis] * rho_batch
        lp_theta_star = log_prob_func(theta_star)

        accept_prob = np.minimum(1.0, np.exp(lp_theta_star - lp_theta))
        accepts = rng.binomial(1, accept_prob).astype(bool)

        theta_new = np.where(accepts[:, np.newaxis], theta_star, theta_batch)

        # Add to storage
        new_idx = theta_storage.shape[1]
        theta_storage = np.concatenate(
            [theta_storage, theta_new[:, :, np.newaxis]], axis=1
        )

        return new_idx, accepts, theta_storage

    def transition(theta_current, theta_storage_start):
        """Single transition for all chains."""
        # Initialize theta storage for this transition
        theta_storage = theta_current[:, :, np.newaxis].copy()

        rho = random_directions()
        theta_idx, accepts, theta_storage = metropolis(
            theta_current, rho, theta_storage
        )

        # Initialize tree with metropolis result
        theta_new = theta_storage[:, :, theta_idx]
        tree = leaf(theta_new)
        tree[:, 0] = theta_idx
        tree[:, 2] = theta_idx
        tree[:, 3] = theta_idx

        depths = np.zeros(num_chains, dtype=int)
        directions = rng.integers(0, 2, size=(num_chains, max_doublings))

        for tree_depth in range(max_doublings):
            direction_batch = directions[:, tree_depth]

            # Build trees for both directions (we'll select later)
            # This is still somewhat sequential but vectorized across chains
            active_forward = direction_batch == 1
            active_backward = direction_batch == 0

            if np.any(active_forward):
                tree_forward, success_f, theta_storage = build_tree(
                    tree_depth, theta_storage, tree, rho, 1
                )
            else:
                tree_forward = None
                success_f = np.zeros(num_chains, dtype=bool)

            if np.any(active_backward):
                tree_backward, success_b, theta_storage = build_tree(
                    tree_depth, theta_storage, tree, rho, 0
                )
            else:
                tree_backward = None
                success_b = np.zeros(num_chains, dtype=bool)

            # Combine based on direction
            if tree_forward is not None and tree_backward is not None:
                tree_next = np.where(
                    active_forward[:, np.newaxis], tree_forward, tree_backward
                )
                success = np.where(active_forward, success_f, success_b)
            elif tree_forward is not None:
                tree_next = tree_forward
                success = success_f
            elif tree_backward is not None:
                tree_next = tree_backward
                success = success_b
            else:
                break

            # Update depths for successful chains
            depths = np.where(success, tree_depth, depths)

            # Update tree only for successful chains
            tree = np.where(success[:, np.newaxis], tree_next, tree)

            # Stop if no chains are active
            if not np.any(success):
                break

            # Check stopping condition
            stopped = ~stopping_condition(tree)
            if np.all(stopped):
                break

        # Extract final thetas
        final_idx = tree[:, 0].astype(int)
        theta_final = theta_storage[np.arange(num_chains), :, final_idx]

        return theta_final, accepts, depths

    # Main sampling loop
    draws = np.zeros((num_chains, num_draws, dim))
    accepts = np.zeros((num_chains, num_draws), dtype=int)
    depths = np.zeros((num_chains, num_draws), dtype=int)

    draws[:, 0, :] = theta_init

    for m in tqdm.tqdm(range(1, num_draws), initial=1, total=num_draws):
        draws[:, m, :], accepts[:, m], depths[:, m] = transition(
            draws[:, m - 1, :], None
        )

    return draws, accepts, depths
