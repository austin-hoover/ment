"""No-Underrun Sampler (NURS).

https://github.com/bob-carpenter/no-underrun-sampler
"""
import math
from collections.abc import Callable

import torch
from tqdm import tqdm
from tqdm import trange

from .core import Sampler


def sample_nurs(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    init_states: torch.Tensor,
    num_samples: int,
    step_size: float = 0.1,
    max_doublings: int = 10,
) -> torch.Tensor:
    """
    No-Underrun Sampler (NURS).

    [From Google Gemini]

    Args:
        log_prob_fn: Callable returning log probabilities for a batch of states.
            Input shape: (batch_size, d), Output shape: (batch_size,)
        init_states: Tensor of shape (n, d) representing starting states for n chains.
        num_samples: Integer, the number of MCMC samples to draw per chain.
        step_size: Float, the distance between evaluated points.
        max_doublings: Integer, the max number of times the orbit can double.

    Returns:
        samples: Tensor of shape (num_samples, n, d) containing the MCMC chains.
    """
    device = init_states.device
    n, d = init_states.shape

    # Pre-allocate tensor to store all samples across all chains
    samples = torch.zeros((num_samples, n, d), device=device)

    current_states = init_states.clone()
    current_log_probs = log_prob_fn(current_states)

    for i in trange(num_samples):
        # 1. Sample uniform directions on the unit sphere for all n chains
        directions = torch.randn(n, d, device=device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)

        # 2. Slice Sampling Setup: Draw n log-uniform height variables
        log_y = current_log_probs - torch.empty(n, device=device).exponential_()

        # 3. Orbit Initialization: Random offsets for all chains
        shifts = torch.rand(n, device=device) * step_size
        left_steps = -shifts
        right_steps = step_size - shifts

        # State tracking for reservoir sampling
        proposed_states = current_states.clone()
        proposed_log_probs = current_log_probs.clone()

        # Valid state counts per chain (the initial state is always valid)
        valid_counts = torch.ones(n, dtype=torch.long, device=device)

        # Mask tracking which chains are still actively expanding
        active_mask = torch.ones(n, dtype=torch.bool, device=device)

        # 4. Batched Constant Velocity Orbit Expansion
        for k in range(max_doublings):
            if not active_mask.any():
                break  # All chains have met the stopping criterion

            # Randomly decide expansion direction for each chain
            expand_right = torch.rand(n, device=device) > 0.5
            num_new = 2**k

            # Generate new step distances
            step_increments = step_size * torch.arange(
                1, num_new + 1, device=device
            ).unsqueeze(0)
            new_steps_right = right_steps.unsqueeze(1) + step_increments
            new_steps_left = left_steps.unsqueeze(1) - step_increments

            # Apply expansion direction to steps
            new_steps = torch.where(
                expand_right.unsqueeze(1), new_steps_right, new_steps_left
            )

            # 5. Parallelized Log-Prob Evaluations
            # Calculate proposals for all n chains and their new points simultaneously
            proposals = current_states.unsqueeze(1) + new_steps.unsqueeze(
                2
            ) * directions.unsqueeze(1)

            # Flatten to evaluate all (n * num_new) states at once
            proposals_flat = proposals.view(n * num_new, d)
            log_probs_flat = log_prob_fn(proposals_flat)
            log_probs = log_probs_flat.view(n, num_new)

            # Filter valid states that are in the slice AND belong to active chains
            in_slice = (log_probs > log_y.unsqueeze(1)) & active_mask.unsqueeze(1)

            # 6. Reservoir Sampling (replaces jagged array filtering)
            new_valid_counts = in_slice.sum(dim=1)
            total_counts = valid_counts + new_valid_counts

            # To simulate uniform sampling among new valid states, assign random scores
            # and pick the highest score. Invalid states get -1.0.
            rand_scores = torch.where(
                in_slice, torch.rand(n, num_new, device=device), -1.0
            )
            chosen_idx = rand_scores.argmax(dim=1)

            new_proposed_states = proposals[torch.arange(n, device=device), chosen_idx]
            new_proposed_log_probs = log_probs[
                torch.arange(n, device=device), chosen_idx
            ]

            # Determine if we should accept the newly drawn state based on reservoir weights
            accept_prob = new_valid_counts.float() / total_counts.float()
            accept_mask = (torch.rand(n, device=device) < accept_prob) & (
                new_valid_counts > 0
            )

            proposed_states = torch.where(
                accept_mask.unsqueeze(1), new_proposed_states, proposed_states
            )
            proposed_log_probs = torch.where(
                accept_mask, new_proposed_log_probs, proposed_log_probs
            )
            valid_counts = total_counts

            # Update left/right boundaries strictly for active chains
            right_steps = torch.where(
                active_mask & expand_right, new_steps[:, -1], right_steps
            )
            left_steps = torch.where(
                active_mask & ~expand_right, new_steps[:, -1], left_steps
            )

            # 7. No-Underrun Condition (Stopping Criterion)
            # Check the endpoints of the currently expanded line segment
            end_pts = torch.stack([left_steps, right_steps], dim=1)
            end_proposals = current_states.unsqueeze(1) + end_pts.unsqueeze(
                2
            ) * directions.unsqueeze(1)
            end_log_probs = log_prob_fn(end_proposals.view(n * 2, d)).view(n, 2)

            # An orbit underruns if both endpoints fall outside the slice threshold
            underrun = (end_log_probs < log_y.unsqueeze(1)).all(dim=1)

            # Deactivate chains that have underrun
            active_mask = active_mask & ~underrun

        # 8. Commit Updates
        current_states = proposed_states
        current_log_probs = proposed_log_probs
        samples[i] = current_states

    return samples


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
        def log_prob_func(x: torch.Tensor) -> torch.Tensor:
            return torch.log(prob_func(x) + 1e-12)

        x = sample_nurs(
            log_prob_func,
            init_states=self.start,
            num_samples=int(math.ceil(size / float(self.chains))),
            step_size=self.step_size,
            max_doublings=self.max_doublings,
        )

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x[:size]
        return x
