import math
from typing import Callable

import torch
import tqdm

from .core import Sampler


@torch.no_grad()
def sample_nurs(
    log_prob_func: Callable[[torch.Tensor], torch.Tensor],
    theta_init: torch.Tensor,
    n_draws: int,
    step_size: float,
    max_doublings: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    n_chains, dim = theta_init.shape
    device = theta_init.device
    dtype = theta_init.dtype

    log_step_size = math.log(step_size)
    log_threshold = math.log(threshold)

    max_points = 2**max_doublings

    draws = torch.empty((n_draws, n_chains, dim), device=device, dtype=dtype)
    accepts = torch.zeros((n_draws, n_chains), device=device, dtype=torch.long)
    depths = torch.zeros((n_draws, n_chains), device=device, dtype=torch.long)

    current_theta = theta_init.clone()
    draws[0] = current_theta

    for i in tqdm.tqdm(range(1, n_draws), initial=1, total=n_draws):
        # 1. Random Direction
        u = torch.randn((n_chains, dim), device=device, dtype=dtype)
        rho = u / torch.linalg.norm(u, dim=1, keepdim=True)

        # 2. Metropolis Shift Step
        lp_curr = log_prob_func(current_theta)
        s = (torch.rand(n_chains, device=device, dtype=dtype) - 0.5) * step_size
        theta_star = current_theta + s[:, None] * rho
        lp_star = log_prob_func(theta_star)

        accept_prob = torch.clip(torch.exp(lp_star - lp_curr), max=1.0)
        accept = torch.rand(n_chains, device=device, dtype=dtype) < accept_prob

        current_theta = torch.where(accept[:, None], theta_star, current_theta)
        lp_curr = torch.where(accept, lp_star, lp_curr)
        accepts[i] = accept.long()

        # 3. Batch Lattice Expansion & Parallel Density Evaluation
        # Pre-generate potential step indices [-max_points, ..., max_points]
        step_offsets = torch.arange(
            -max_points, max_points + 1, device=device, dtype=dtype
        )

        # Batch evaluation across all chains and potential points: (n_chains, 2*max_points + 1, dim)
        candidate_points = (
            current_theta[:, None, :]
            + (step_offsets[None, :, None] * step_size) * rho[:, None, :]
        )

        # Reshape to batch evaluate through user's target function
        flat_candidates = candidate_points.view(-1, dim)
        flat_logp = log_prob_func(flat_candidates)
        logp_grid = flat_logp.view(n_chains, 2 * max_points + 1)

        # Center index corresponds to index `max_points` (offset 0)
        center_idx = max_points

        # Track active tree bounds for each chain
        left_idx = torch.full((n_chains,), center_idx, device=device, dtype=torch.long)
        right_idx = torch.full((n_chains,), center_idx, device=device, dtype=torch.long)

        # Current combined tree state
        tree_logp_sum = logp_grid[torch.arange(n_chains), center_idx]
        selected_theta = current_theta.clone()

        chain_stopped = torch.zeros(n_chains, device=device, dtype=torch.bool)
        directions = torch.randint(0, 2, size=(n_chains, max_doublings), device=device)

        final_depths = torch.zeros(n_chains, device=device, dtype=torch.long)

        for depth in range(max_doublings):
            active_mask = ~chain_stopped
            if not active_mask.any():
                break

            dir_d = directions[:, depth]  # 1 = right, 0 = left
            tree_size = 2**depth

            # Calculate prospective new sub-tree indices
            new_left = torch.where(dir_d == 1, right_idx + 1, left_idx - tree_size)
            new_right = torch.where(dir_d == 1, right_idx + tree_size, left_idx - 1)

            # Compute logp sums for proposed extension
            # Gather sub-tree log-probabilities
            idx_range = torch.arange(tree_size, device=device)
            ext_indices = new_left[:, None] + idx_range[None, :]
            ext_logp = torch.gather(logp_grid, 1, ext_indices)
            ext_logp_sum = torch.logsumexp(ext_logp, dim=1)

            # Categorical tree combination / update state
            comb_logp = torch.logsumexp(
                torch.stack([tree_logp_sum, ext_logp_sum], dim=0), dim=0
            )
            prob_ext = torch.exp(ext_logp_sum - comb_logp)

            # Select new state conditionally
            choose_ext = torch.rand(n_chains, device=device) < prob_ext

            # Pick state from extension subtree
            ext_probs = torch.softmax(ext_logp, dim=1)
            ext_selected_offset = torch.multinomial(ext_probs, 1).squeeze(-1)
            ext_selected_idx = new_left + ext_selected_offset

            # Update selected position for active chains
            ext_theta = (
                current_theta
                + (ext_selected_idx - center_idx)[:, None] * step_size * rho
            )
            update_mask = active_mask & choose_ext
            selected_theta = torch.where(
                update_mask[:, None], ext_theta, selected_theta
            )

            # Update overall tree boundaries & logsumexp
            tree_logp_sum = torch.where(active_mask, comb_logp, tree_logp_sum)
            left_idx = torch.where(active_mask & (dir_d == 0), new_left, left_idx)
            right_idx = torch.where(active_mask & (dir_d == 1), new_right, right_idx)

            # Stopping Condition Check: max(lp_left, lp_right) <= log_threshold + log_step + tree_logp
            lp_l = logp_grid[torch.arange(n_chains), left_idx]
            lp_r = logp_grid[torch.arange(n_chains), right_idx]
            log_eps = log_threshold + log_step_size + tree_logp_sum

            stop = (lp_l < log_eps) & (lp_r < log_eps)
            chain_stopped = chain_stopped | stop
            final_depths = torch.where(
                active_mask, torch.tensor(depth, device=device), final_depths
            )

        current_theta = selected_theta
        draws[i] = current_theta
        depths[i] = final_depths

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
        return x[:size]
