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

        # 3. Dynamic Tree Initialization
        tree_logp_sum = lp_curr.clone()
        selected_theta = current_theta.clone()

        # Track relative offsets from current_theta (start at 0)
        left_offset = torch.zeros(n_chains, device=device, dtype=torch.long)
        right_offset = torch.zeros(n_chains, device=device, dtype=torch.long)

        # We also need to track the actual log_prob at the exact left and right boundaries
        lp_left = lp_curr.clone()
        lp_right = lp_curr.clone()

        chain_stopped = torch.zeros(n_chains, device=device, dtype=torch.bool)
        directions = torch.randint(0, 2, size=(n_chains, max_doublings), device=device)
        final_depths = torch.zeros(n_chains, device=device, dtype=torch.long)

        for depth in range(max_doublings):
            active_mask = ~chain_stopped
            active_indices = active_mask.nonzero(as_tuple=True)[0]

            if len(active_indices) == 0:
                break

            n_active = len(active_indices)
            tree_size = 2**depth

            # Extract states just for active chains
            act_dir = directions[active_indices, depth]
            act_left_off = left_offset[active_indices]
            act_right_off = right_offset[active_indices]
            act_curr_theta = current_theta[active_indices]
            act_rho = rho[active_indices]

            # Compute new indices for the extension tree
            # Shape: (n_active, tree_size)
            idx_range = torch.arange(tree_size, device=device, dtype=torch.long)

            new_offsets = torch.where(
                (act_dir == 1)[:, None],
                act_right_off[:, None] + 1 + idx_range[None, :],
                act_left_off[:, None] - tree_size + idx_range[None, :],
            )

            # Evaluate log_prob only for new points of active chains
            new_points = (
                act_curr_theta[:, None, :]
                + (new_offsets[:, :, None] * step_size) * act_rho[:, None, :]
            )
            flat_new_points = new_points.view(-1, dim)
            flat_ext_logp = log_prob_func(flat_new_points)
            ext_logp = flat_ext_logp.view(n_active, tree_size)

            # Logsumexp of the extension
            act_ext_logp_sum = torch.logsumexp(ext_logp, dim=1)

            # Update combined tree logp
            act_comb_logp = torch.logsumexp(
                torch.stack([tree_logp_sum[active_indices], act_ext_logp_sum], dim=0),
                dim=0,
            )

            # Select new states from the extension block
            prob_ext = torch.exp(act_ext_logp_sum - act_comb_logp)
            choose_ext = torch.rand(n_active, device=device) < prob_ext

            ext_probs = torch.softmax(ext_logp, dim=1)
            ext_selected_local_idx = torch.multinomial(ext_probs, 1).squeeze(-1)
            ext_selected_offset = torch.gather(
                new_offsets, 1, ext_selected_local_idx[:, None]
            ).squeeze(-1)

            act_ext_theta = (
                act_curr_theta + ext_selected_offset[:, None] * step_size * act_rho
            )

            # Write back state updates safely to full tensors
            # 1. Update Selected Theta
            update_mask = torch.zeros(n_chains, device=device, dtype=torch.bool)
            update_mask[active_indices] = choose_ext

            selected_theta = torch.where(
                update_mask[:, None],
                torch.zeros_like(selected_theta).scatter_(
                    0, active_indices[:, None].expand(-1, dim), act_ext_theta
                ),
                selected_theta,
            )

            # 2. Update Tree Boundaries and Logps
            tree_logp_sum[active_indices] = act_comb_logp

            # Boundaries: if dir==0, update left. If dir==1, update right.
            left_update_mask = active_mask & (directions[:, depth] == 0)
            right_update_mask = active_mask & (directions[:, depth] == 1)

            # The boundary values are always at index 0 (left-most) or -1 (right-most) of the extension block
            new_left_bound_lp = ext_logp[:, 0]
            new_right_bound_lp = ext_logp[:, -1]

            left_offset[active_indices] = torch.where(
                act_dir == 0, new_offsets[:, 0], act_left_off
            )
            right_offset[active_indices] = torch.where(
                act_dir == 1, new_offsets[:, -1], act_right_off
            )

            lp_left[active_indices] = torch.where(
                act_dir == 0, new_left_bound_lp, lp_left[active_indices]
            )
            lp_right[active_indices] = torch.where(
                act_dir == 1, new_right_bound_lp, lp_right[active_indices]
            )

            # 3. Stopping Condition Check for active chains
            log_eps = log_threshold + log_step_size + tree_logp_sum[active_indices]
            stop = (lp_left[active_indices] < log_eps) & (
                lp_right[active_indices] < log_eps
            )

            chain_stopped[active_indices] = stop
            final_depths[active_indices] = depth

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
