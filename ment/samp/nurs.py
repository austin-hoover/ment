"""No-Underrun Sampler (NURS).

https://github.com/bob-carpenter/no-underrun-sampler
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.special
import tqdm

from .core import Sampler


@dataclass
class Tree:
    theta: np.ndarray
    logp: np.ndarray
    theta_l: np.ndarray
    theta_r: np.ndarray
    logp_l: np.ndarray
    logp_r: np.ndarray


def sample_nurs(
    rng: np.random.Generator,
    log_prob_func: Callable,
    theta_init: np.ndarray,
    n_draws: int,
    step_size: float,
    max_doublings: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized NURS sampler for multiple parallel chains.

    Args:
        log_prob_func: Returns log probability density (up to constant).
            Accepts array of shape (n_chains, n_dim). Returns array
            of shape (n_chains,)
    """
    n_chains, dim = theta_init.shape
    log_step_size = np.log(step_size)
    log_threshold = np.log(threshold)

    def stopping_condition(tree):
        log_epsilon = log_threshold + log_step_size + tree.logp
        return (tree.logp_l < log_epsilon) & (tree.logp_r < log_epsilon)

    def leaf(theta: np.ndarray):
        logp = log_prob_func(theta)
        return Tree(
            theta=theta,
            logp=logp,
            theta_l=theta,
            theta_r=theta,
            logp_l=logp,
            logp_r=logp,
        )

    def combine_trees(tree1: Tree, tree2: Tree, direction: np.ndarray) -> Tree:
        logp1 = tree1.logp
        logp2 = tree2.logp
        logp12 = scipy.special.logsumexp(np.stack([logp1, logp2], axis=0), axis=0)
        probs = np.exp(logp2 - logp12)
        update = rng.random(n_chains) < probs
        theta = np.where(update[:, None], tree2.theta, tree1.theta)

        cond = direction == 1
        theta_l = np.where(cond[:, None], tree1.theta_l, tree2.theta_l)
        theta_r = np.where(cond[:, None], tree2.theta_r, tree1.theta_r)
        logp_l = np.where(cond, tree1.logp_l, tree2.logp_l)
        logp_r = np.where(cond, tree2.logp_r, tree1.logp_r)

        return Tree(
            theta=theta,
            logp=logp12,
            theta_l=theta_l,
            theta_r=theta_r,
            logp_l=logp_l,
            logp_r=logp_r,
        )

    def build_tree(
        depth: int, theta_last: np.ndarray, rho: np.ndarray, direction: np.ndarray
    ) -> Tree | None:
        h = step_size * (2 * direction - 1)
        if depth == 0:
            theta_next = theta_last + h[:, None] * rho
            return leaf(theta_next)
        tree1 = build_tree(depth - 1, theta_last, rho, direction)
        if tree1 is None:
            return None
        theta_mid = np.where((direction == 1)[:, None], tree1.theta_r, tree1.theta_l)
        tree2 = build_tree(depth - 1, theta_mid, rho, direction)
        if tree2 is None:
            return None
        tree = combine_trees(tree1, tree2, direction)
        stop = stopping_condition(tree)
        if np.any(stop):
            return None
        return tree

    def random_direction() -> np.ndarray:
        u = rng.normal(size=(n_chains, dim))
        return u / np.linalg.norm(u, axis=1, keepdims=True)

    def metropolis(theta: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lp_theta = log_prob_func(theta)
        s = (rng.random(n_chains) - 0.5) * step_size
        theta_star = theta + s[:, None] * rho
        lp_theta_star = log_prob_func(theta_star)
        accept_prob = np.minimum(1.0, np.exp(lp_theta_star - lp_theta))
        accept = rng.random(n_chains) < accept_prob
        new_theta = np.where(accept[:, None], theta_star, theta)
        return new_theta, accept

    def transition(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        rho = random_direction()
        theta, accept = metropolis(theta, rho)
        tree = leaf(theta)
        directions = rng.integers(0, 2, size=(n_chains, max_doublings))

        for tree_depth in range(max_doublings):
            direction = directions[:, tree_depth]
            theta_mid = np.where((direction == 1)[:, None], tree.theta_r, tree.theta_l)
            tree_next = build_tree(tree_depth, theta_mid, rho, direction)
            if tree_next is None:
                break
            tree = combine_trees(tree, tree_next, direction)
            stop = stopping_condition(tree)
            if np.any(stop):
                break

        return (tree.theta, accept, tree_depth)

    draws = np.zeros((n_draws, n_chains, dim))
    accepts = np.zeros((n_draws, n_chains), dtype=int)
    depths = np.zeros((n_draws, n_chains), dtype=int)

    draws[0] = theta_init
    for m in tqdm.tqdm(range(1, n_draws), initial=1, total=n_draws):
        draws[m], accepts[m], depths[m] = transition(draws[m - 1])
    return draws, accepts, depths


class NURSSampler(Sampler):
    def __init__(self) -> None:
        raise NotImplementedError()
