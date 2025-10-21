"""No-Underrun Sampler (NURS).

https://github.com/bob-carpenter/no-underrun-sampler
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import tqdm

from .core import Sampler


@dataclass
class Tree:
    theta: torch.Tensor
    logp: torch.Tensor
    theta_l: torch.Tensor
    theta_r: torch.Tensor
    logp_l: torch.Tensor
    logp_r: torch.Tensor


def sample_nurs(
    log_prob_func: Callable,
    theta_init: torch.Tensor,
    n_draws: int,
    step_size: float,
    max_doublings: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized NURS sampler for multiple parallel chains.

    Args:
        log_prob_func: Returns log probability density (up to constant).
            Accepts array of shape (n_chains, n_dim). Returns array
            of shape (n_chains,)
        theta_init: Starting points, shape (n_chains, n_dim).
    """
    rng = torch.Generator()

    n_chains, dim = theta_init.shape
    log_step_size = torch.log(torch.as_tensor(step_size))
    log_threshold = torch.log(torch.as_tensor(threshold))

    def stopping_condition(tree: Tree) -> bool:
        log_epsilon = log_threshold + log_step_size + tree.logp
        return (tree.logp_l < log_epsilon) & (tree.logp_r < log_epsilon)

    def leaf(theta: torch.Tensor):
        logp = log_prob_func(theta)
        return Tree(
            theta=theta,
            logp=logp,
            theta_l=theta,
            theta_r=theta,
            logp_l=logp,
            logp_r=logp,
        )

    def combine_trees(tree1: Tree, tree2: Tree, direction: torch.Tensor) -> Tree:
        logp1 = tree1.logp
        logp2 = tree2.logp
        logp12 = torch.logsumexp(torch.stack([logp1, logp2], axis=0), axis=0)
        probs = torch.exp(logp2 - logp12)
        update = torch.rand(n_chains, generator=rng) < probs
        theta = torch.where(update[:, None], tree2.theta, tree1.theta)

        cond = direction == 1
        theta_l = torch.where(cond[:, None], tree1.theta_l, tree2.theta_l)
        theta_r = torch.where(cond[:, None], tree2.theta_r, tree1.theta_r)
        logp_l = torch.where(cond, tree1.logp_l, tree2.logp_l)
        logp_r = torch.where(cond, tree2.logp_r, tree1.logp_r)

        return Tree(
            theta=theta,
            logp=logp12,
            theta_l=theta_l,
            theta_r=theta_r,
            logp_l=logp_l,
            logp_r=logp_r,
        )

    def build_tree(
        depth: int, theta_last: torch.Tensor, rho: torch.Tensor, direction: torch.Tensor
    ) -> Tree | None:

        h = step_size * (2 * direction - 1)
        if depth == 0:
            theta_next = theta_last + h[:, None] * rho
            return leaf(theta_next)

        tree1 = build_tree(depth - 1, theta_last, rho, direction)
        if tree1 is None:
            return None

        theta_mid = torch.where((direction == 1)[:, None], tree1.theta_r, tree1.theta_l)
        tree2 = build_tree(depth - 1, theta_mid, rho, direction)
        if tree2 is None:
            return None

        tree = combine_trees(tree1, tree2, direction)
        stop = stopping_condition(tree)
        if torch.any(stop):
            return None

        return tree

    def random_direction() -> torch.Tensor:
        u = torch.randn((n_chains, dim), generator=rng)
        return u / torch.linalg.norm(u, axis=1, keepdims=True)

    def metropolis(
        theta: torch.Tensor, rho: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lp_theta = log_prob_func(theta)
        s = (torch.rand(n_chains, generator=rng) - 0.5) * step_size
        theta_star = theta + s[:, None] * rho
        lp_theta_star = log_prob_func(theta_star)
        accept_prob = torch.minimum(
            torch.tensor(1.0), torch.exp(lp_theta_star - lp_theta)
        )
        accept = torch.rand(n_chains, generator=rng) < accept_prob
        new_theta = torch.where(accept[:, None], theta_star, theta)
        return new_theta, accept

    def transition(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        rho = random_direction()
        theta, accept = metropolis(theta, rho)
        tree = leaf(theta)
        directions = torch.randint(
            low=0, high=2, size=(n_chains, max_doublings), generator=rng
        )

        for tree_depth in range(max_doublings):
            direction = directions[:, tree_depth]
            theta_mid = torch.where(
                (direction == 1)[:, None], tree.theta_r, tree.theta_l
            )
            tree_next = build_tree(tree_depth, theta_mid, rho, direction)
            if tree_next is None:
                break
            tree = combine_trees(tree, tree_next, direction)
            stop = stopping_condition(tree)
            if torch.any(stop):
                break

        return (tree.theta, accept, tree_depth)

    draws = torch.zeros((n_draws, n_chains, dim))
    accepts = torch.zeros((n_draws, n_chains), dtype=int)
    depths = torch.zeros((n_draws, n_chains), dtype=int)

    draws[0] = theta_init
    for i in tqdm.tqdm(range(1, n_draws), initial=1, total=n_draws):
        draws[i], accepts[i], depths[i] = transition(draws[i - 1])
    return draws, accepts, depths


def sample_nurs_ssa(
    log_prob_func: Callable,
    theta_init: torch.Tensor,
    n_draws: int,
    min_step_size: float,
    max_step_doublings: int,
    max_tree_doublings: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    rng = torch.Generator()

    n_chains, dim = theta_init.shape
    log_threshold = torch.log(torch.as_tensor(threshold))

    def stopping_condition(tree: Tree, current_step_size: float) -> bool:
        log_epsilon = log_threshold + math.log(current_step_size) + tree.logp
        return (tree.logp_l < log_epsilon) & (tree.logp_r < log_epsilon)

    def leaf(theta: torch.Tensor):
        logp = log_prob_func(theta)
        return Tree(
            theta=theta,
            logp=logp,
            theta_l=theta,
            theta_r=theta,
            logp_l=logp,
            logp_r=logp,
        )

    def combine_trees(tree1: Tree, tree2: Tree, direction: torch.Tensor) -> Tree:
        logp1 = tree1.logp
        logp2 = tree2.logp
        logp12 = torch.logsumexp(torch.stack([logp1, logp2], axis=0), axis=0)
        probs = torch.exp(logp2 - logp12)
        update = torch.rand(n_chains, generator=rng) < probs
        theta = torch.where(update[:, None], tree2.theta, tree1.theta)

        cond = direction == 1
        theta_l = torch.where(cond[:, None], tree1.theta_l, tree2.theta_l)
        theta_r = torch.where(cond[:, None], tree2.theta_r, tree1.theta_r)
        logp_l = torch.where(cond, tree1.logp_l, tree2.logp_l)
        logp_r = torch.where(cond, tree2.logp_r, tree1.logp_r)

        return Tree(
            theta=theta,
            logp=logp12,
            theta_l=theta_l,
            theta_r=theta_r,
            logp_l=logp_l,
            logp_r=logp_r,
        )

    def build_tree(
        depth: int,
        theta_last: torch.Tensor,
        rho: torch.Tensor,
        direction: torch.Tensor,
        current_step_size: float,
    ) -> Tree | None:

        h = current_step_size * (2 * direction - 1)
        if depth == 0:
            theta_next = theta_last + h[:, None] * rho
            return leaf(theta_next)

        tree1 = build_tree(depth - 1, theta_last, rho, direction, current_step_size)
        if tree1 is None:
            return None

        theta_mid = torch.where((direction == 1)[:, None], tree1.theta_r, tree1.theta_l)
        tree2 = build_tree(depth - 1, theta_mid, rho, direction, current_step_size)
        if tree2 is None:
            return None

        tree = combine_trees(tree1, tree2, direction)
        stop = stopping_condition(tree, current_step_size)
        if torch.any(stop):
            return None

        return tree

    def random_direction() -> torch.Tensor:
        u = torch.randn((n_chains, dim), generator=rng)
        return u / torch.linalg.norm(u, axis=1, keepdims=True)

    def metropolis(
        theta: torch.Tensor,
        rho: torch.Tensor,
        current_step_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lp_theta = log_prob_func(theta)
        s = (torch.rand(n_chains, generator=rng) - 0.5) * current_step_size
        theta_star = theta + s[:, None] * rho
        lp_theta_star = log_prob_func(theta_star)
        accept_prob = torch.minimum(
            torch.tensor(1.0), torch.exp(lp_theta_star - lp_theta)
        )
        accept = torch.rand(n_chains, generator=rng) < accept_prob
        new_theta = torch.where(accept[:, None], theta_star, theta)
        return new_theta, accept

    def transition(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        rho = random_direction()
        directions = torch.randint(
            low=0, high=2, size=(n_chains, max_tree_doublings), generator=rng
        )
        current_step_size = min_step_size
        for _ in range(max_step_doublings):
            theta, accept = metropolis(theta, rho, current_step_size)
            tree = leaf(theta)
            for tree_depth in range(max_tree_doublings):
                direction = directions[:, tree_depth]
                theta_mid = torch.where(
                    (direction == 1)[:, None], tree.theta_r, tree.theta_l
                )
                tree_next = build_tree(
                    tree_depth, theta_mid, rho, direction, current_step_size
                )
                if not tree_next:
                    break
                tree = combine_trees(tree, tree_next, direction)
                stop = stopping_condition(tree, current_step_size)
                if torch.any(stop):
                    return (tree.theta, accept, tree_depth)

            current_step_size *= 2.0

        return (tree.theta, accept, tree_depth)

    draws = torch.zeros((n_draws, n_chains, dim))
    accepts = torch.zeros((n_draws, n_chains), dtype=int)
    depths = torch.zeros((n_draws, n_chains), dtype=int)

    draws[0] = theta_init
    for i in tqdm.tqdm(range(1, n_draws), initial=1, total=n_draws):
        draws[i], accepts[i], depths[i] = transition(draws[i - 1])
    return draws, accepts, depths


class NURSSampler(Sampler):
    def __init__(
        self,
        start: torch.Tensor,
        step_size: float = 0.2,
        max_doublings: int = 10,
        threshold: float = 1e-5,
        **kws,
    ) -> None:
        super().__init__(**kws)
        self.start = start
        self.chains = start.shape[0]
        self.step_size = step_size
        self.max_doublings = max_doublings
        self.threshold = threshold

    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        size_per_chain = int(math.ceil(size / float(self.chains)))

        def log_prob_func(x: torch.Tensor) -> torch.Tensor:
            return torch.log(prob_func(x) + 1e-12)

        x, _, _ = sample_nurs(
            log_prob_func=log_prob_func,
            theta_init=self.start,
            n_draws=size_per_chain,
            step_size=self.step_size,
            max_doublings=self.max_doublings,
            threshold=self.threshold,
        )
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x[:size]
        return x
