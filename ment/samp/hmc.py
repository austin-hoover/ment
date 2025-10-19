"""Hamiltonian Monte Carlo (HCM)

Adapted from https://github.com/AdamCobb/hamiltorch.
"""
import math
from typing import Callable

import torch

from .core import Sampler
from ..utils import wrap_tqdm


def compute_gradients_loop(log_prob_func: Callable, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_()
    log_prob = log_prob_func(x)
    for i in range(x.shape[0]):
        log_prob[i].backward(retain_graph=True)
    return x.grad


def compute_gradients(log_prob_func: Callable, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_()
    log_prob = log_prob_func(x)
    log_prob.sum().backward()
    return x.grad


def resample_momentum(x: torch.Tensor, cov_matrix: torch.Tensor = None) -> torch.Tensor:
    mean = torch.zeros(x.shape[1])
    if cov_matrix is None:
        cov_matrix = torch.eye(x.shape[1])
    dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
    return dist.sample((x.shape[0],))


def integrate_leapfrog(
    x: torch.Tensor,
    p: torch.Tensor,
    log_prob_func: Callable,
    steps: int = 10,
    step_size: float = 0.1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Propose new set of parameters and momentum.

    Parameters
    ----------
    x : torch.tensor
        Parameter array, shape (..., D).
    p : torch.tensor
        Momentum array, shape (..., D).
    log_prob_func : function
        Vectorized log-probability function.
    steps : int
        Number of steps per trajectory (often referred to as L).
    step_size : float
        Integration step size.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC.
        Choices = {
            Sampler.HMC,
            Sampler.HMC_NUTS,
        }

    Returns
    -------
    ret_params : list[torch.Tensor]
        List of parameters collected in the trajectory.
    ret_momentum : list[torch.Tensor]
        List of momentum collected in the trajectory.
    """
    x = torch.clone(x)
    p = torch.clone(p)

    xs, ps = [], []

    p += 0.5 * step_size * compute_gradients(log_prob_func, x)
    for _ in range(steps):
        x += step_size * p
        p += step_size * compute_gradients(log_prob_func, x)
        xs.append(torch.clone(x.detach()))
        ps.append(torch.clone(p.detach()))
    # Only need last for Hamiltonian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
    ps[-1] -= 0.5 * step_size * compute_gradients(log_prob_func, x)

    return xs, ps


def hamiltonian(
    x: torch.Tensor, p: torch.Tensor, log_prob_func: Callable
) -> torch.Tensor:
    potential = -log_prob_func(x)
    kinetic = 0.5 * torch.sum(torch.square(p), axis=1)
    return potential + kinetic


def sample(
    log_prob_func: Callable,
    start: torch.Tensor,
    size: int = 10,
    steps_per_samp: int = 10,
    step_size: float = 0.1,
    burnin: int = 1,
    verbose: int = 0,
) -> dict:
    """Vectorized Hamiltonian Monte Carlo.

    Parameters
    ----------
    log_prob_func : Callable
        Vectorized log-probability function.
    start : torch.Tensor
        Initial particle coordinates. If shape=(ndim,), run a single chain;
        otherwise if shape=(nchain, ndim), run multiple chains.
    size : int
        Number of samples to generate. The number of HMC steps is determined from the
        number of chains.
    steps_per_samp : int
        Number of integration steps per trajectory.
    step_size : float
        Size of each step to take when doing the numerical integration.
    burnin : int
        Number of steps before samples are collected.
    verbose : int
        If 0, do not display progress bar.

    Returns
    -------
    list[torch.Tensor]
        A list of parameter samples.
    """
    device = start.device

    if burnin >= size:
        raise RuntimeError("burn must be less than size.")

    if start.ndim == 1:
        start = start[None, :]

    x = start.clone()
    samples = [x.clone().detach()]

    n_chains = start.shape[0]
    n_steps = int(math.ceil(size / n_chains))
    n_accepted = 0
    acceptance_rate = None

    for n in wrap_tqdm(range(n_steps + burnin), verbose):
        # Push particles
        p = resample_momentum(x)
        ham = hamiltonian(x, p, log_prob_func)
        x_traj, p_traj = integrate_leapfrog(
            x,
            p,
            log_prob_func,
            steps=steps_per_samp,
            step_size=step_size,
        )

        # MH proposal
        x_new = x_traj[-1].to(device).detach()
        p_new = p_traj[-1].to(device).detach()
        ham_new = hamiltonian(x_new, p_new, log_prob_func)

        rho = torch.minimum(torch.zeros(n_chains), ham - ham_new)
        accept = rho >= torch.log(torch.rand(n_chains))
        if n >= burnin:
            n_accepted += torch.sum(accept)
            acceptance_rate = n_accepted / ((n + 1 - burnin) * n_chains)

        if verbose > 1:
            print("Step: {},".format(n))
            print("Hamiltonian current : {},".format(ham))
            print("Hamiltonian proposed: {},".format(ham_new))

        x[accept] = x_new[accept]
        samples.append(x.detach().clone())

    samples = samples[burnin:]
    samples = torch.vstack(samples)
    samples = samples[:size]
    samples = torch.squeeze(samples)

    results = {}
    results["samples"] = samples
    results["acceptance_rate"] = acceptance_rate
    return results


class HamiltonianMonteCarloSampler(Sampler):
    """Hamiltonian Monte Carlo (HMC) sampler."""

    def __init__(
        self,
        chains: int = 1,
        burnin: int = 0,
        start: torch.Tensor = None,
        step_size: float = 0.20,
        steps_per_samp: int = 5,
        **kws,
    ) -> None:
        super().__init__(**kws)

        self.chains = chains
        if self.chains > 1:
            raise NotImplementedError

        self.start = start
        self.burnin = burnin
        self.step_size = step_size
        self.steps_per_samp = steps_per_samp

    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        def log_prob_func(x: torch.Tensor) -> torch.Tensor:
            return torch.log(prob_func(x) + 1.00e-12)

        start = self.start
        if start is None:
            start = torch.zeros(self.ndim)

        results = sample(
            log_prob_func,
            start=start,
            size=size,
            steps_per_samp=self.steps_per_samp,
            step_size=self.step_size,
            burnin=self.burnin,
            verbose=self.verbose,
        )
        self.results = results
        samples = self.results.pop("samples")
        return samples
