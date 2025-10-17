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

    def stopping_condition(trees):
        """Check stopping condition for batch of trees.

        Trees structure: dict with keys 'selected', 'lp', 'left', 'right', 'left_min', 'right_min'
        All values are arrays of shape (num_chains, dim) for theta or (num_chains,) for scalars
        """
        log_epsilon = log_threshold + log_step_size + trees["lp"]
        return (trees["left_min"] < log_epsilon) & (trees["right_min"] < log_epsilon)

    def leaf(theta_batch: np.ndarray):
        """Create leaf nodes for batch of thetas.

        Returns: dict with tree information
        """
        lp = log_prob_func(theta_batch)  # (num_chains,)
        return {
            "selected": theta_batch.copy(),
            "lp": lp,
            "left": theta_batch.copy(),
            "right": theta_batch.copy(),
            "left_min": lp,
            "right_min": lp,
        }

    def combine_trees(tree1, tree2, direction):
        """Combine two trees. Direction: 1=forward, 0=backward."""
        lp1 = tree1["lp"]
        lp2 = tree2["lp"]
        lp12 = logsumexp(np.column_stack([lp1, lp2]), axis=1)

        # Randomly select from tree1 or tree2
        probs = np.exp(lp2 - lp12)
        updates = rng.binomial(1, probs).astype(bool)
        selected = np.where(
            updates[:, np.newaxis], tree2["selected"], tree1["selected"]
        )

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

    def build_tree(depth, tree_current, rho, direction):
        """Build tree for all chains simultaneously.

        Returns: (tree, active_mask) where active_mask indicates which chains succeeded
        """
        h = step_size * (2 * direction - 1)

        if depth == 0:
            # Get theta from current tree's right or left boundary
            if direction == 1:
                theta_last = tree_current["right"]
            else:
                theta_last = tree_current["left"]

            theta_next = theta_last + h * rho
            tree_new = leaf(theta_next)
            active = np.ones(num_chains, dtype=bool)
            return tree_new, active

        # Recursive case
        tree1, active1 = build_tree(depth - 1, tree_current, rho, direction)

        # For chains that failed, return dummy tree
        if not np.any(active1):
            return tree1, active1

        # Update tree_current boundaries for next recursive call
        tree_mid = tree1.copy()

        tree2, active2 = build_tree(depth - 1, tree_mid, rho, direction)

        active = active1 & active2

        if not np.any(active):
            return tree1, active

        tree = combine_trees(tree1, tree2, direction)

        # Check stopping condition
        stopped = ~stopping_condition(tree)
        active = active & ~stopped

        return tree, active

    def random_directions():
        """Generate random unit directions for all chains."""
        u = rng.normal(size=(num_chains, dim))
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        return u / norms

    def metropolis(theta_batch, rho_batch):
        """Vectorized Metropolis step."""
        lp_theta = log_prob_func(theta_batch)
        s = (rng.random(size=num_chains) - 0.5) * step_size
        theta_star = theta_batch + s[:, np.newaxis] * rho_batch
        lp_theta_star = log_prob_func(theta_star)

        accept_prob = np.minimum(1.0, np.exp(lp_theta_star - lp_theta))
        accepts = rng.binomial(1, accept_prob).astype(bool)

        theta_new = np.where(accepts[:, np.newaxis], theta_star, theta_batch)

        return theta_new, accepts

    def transition(theta_current):
        """Single transition for all chains."""
        rho = random_directions()
        theta_metro, accepts = metropolis(theta_current, rho)

        # Initialize tree with metropolis result
        tree = leaf(theta_metro)

        depths = np.zeros(num_chains, dtype=int)
        directions = rng.integers(0, 2, size=(num_chains, max_doublings))

        for tree_depth in range(max_doublings):
            direction_batch = directions[:, tree_depth]

            # We need to handle each direction separately since they modify the tree differently
            # For simplicity, let's process all chains with the same logic but use their individual directions

            # Split by direction
            forward_mask = direction_batch == 1
            backward_mask = direction_batch == 0

            all_active = np.zeros(num_chains, dtype=bool)
            tree_next = {
                k: v.copy() if isinstance(v, np.ndarray) else v for k, v in tree.items()
            }

            # Process forward direction chains
            if np.any(forward_mask):
                tree_f, active_f = build_tree(tree_depth, tree, rho, 1)
                # Update only forward chains
                for key in tree_next:
                    if isinstance(tree_next[key], np.ndarray):
                        if tree_next[key].ndim == 2:  # theta arrays
                            tree_next[key] = np.where(
                                forward_mask[:, np.newaxis] & active_f[:, np.newaxis],
                                tree_f[key],
                                tree_next[key],
                            )
                        else:  # scalar arrays
                            tree_next[key] = np.where(
                                forward_mask & active_f, tree_f[key], tree_next[key]
                            )
                all_active |= forward_mask & active_f

            # Process backward direction chains
            if np.any(backward_mask):
                tree_b, active_b = build_tree(tree_depth, tree, rho, 0)
                # Update only backward chains
                for key in tree_next:
                    if isinstance(tree_next[key], np.ndarray):
                        if tree_next[key].ndim == 2:  # theta arrays
                            tree_next[key] = np.where(
                                backward_mask[:, np.newaxis] & active_b[:, np.newaxis],
                                tree_b[key],
                                tree_next[key],
                            )
                        else:  # scalar arrays
                            tree_next[key] = np.where(
                                backward_mask & active_b, tree_b[key], tree_next[key]
                            )
                all_active |= backward_mask & active_b

            # Update depths for chains that successfully expanded
            depths = np.where(all_active, tree_depth, depths)

            # Combine with previous tree
            tree = combine_trees(
                tree, tree_next, 1
            )  # direction doesn't matter here since we already updated

            # Actually, we should just use tree_next as the new tree for active chains
            for key in tree:
                if isinstance(tree[key], np.ndarray):
                    if tree[key].ndim == 2:
                        tree[key] = np.where(
                            all_active[:, np.newaxis], tree_next[key], tree[key]
                        )
                    else:
                        tree[key] = np.where(all_active, tree_next[key], tree[key])

            # Stop if no chains are active
            if not np.any(all_active):
                break

            # Check stopping condition for all chains
            stopped = ~stopping_condition(tree)
            if np.all(stopped):
                break

        return tree["selected"], accepts.astype(int), depths

    # Main sampling loop
    draws = np.zeros((num_chains, num_draws, dim))
    accepts = np.zeros((num_chains, num_draws), dtype=int)
    depths = np.zeros((num_chains, num_draws), dtype=int)

    draws[:, 0, :] = theta_init

    for m in tqdm.tqdm(range(1, num_draws), initial=1, total=num_draws):
        draws[:, m, :], accepts[:, m], depths[:, m] = transition(draws[:, m - 1, :])

    return draws, accepts, depths
