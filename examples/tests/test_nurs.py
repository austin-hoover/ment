"""Test vectorized NURS sampler on 2D distribution."""

import argparse
import os
import pathlib
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment


plt.style.use("../style.mplstyle")


def log_prob_func_ring(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return torch.sin(torch.pi * x1) - 2.0 * (x1**2 + x2**2 - 2.0) ** 2


def log_prob_func_normal(x: torch.Tensor) -> float:
    return -0.5 * torch.sum(x**2, axis=1)


def get_log_prob_func(name: str) -> Callable:
    if name == "ring":
        return log_prob_func_ring
    if name == "normal":
        return log_prob_func_normal
    return None


def plot_samples(
    log_prob_func: Callable, x_pred: torch.Tensor, xmax: float = 3.0, nbins: int = 64
) -> tuple:
    grid_limits = 2 * [(-xmax, xmax)]
    grid_shape = (nbins, nbins)
    grid_edges = [
        torch.linspace(grid_limits[i][0], grid_limits[i][1], grid_shape[i] + 1)
        for i in range(len(grid_shape))
    ]
    grid_coords = [0.5 * (e[:-1] + e[1:]) for e in grid_edges]
    grid_points = torch.stack(
        [c.ravel() for c in torch.meshgrid(*grid_coords, indexing="ij")], axis=-1
    )

    grid_values_true = torch.exp(log_prob_func(grid_points))
    grid_values_true = grid_values_true.reshape(grid_shape)
    grid_values_true /= torch.sum(grid_values_true)

    grid_values_pred = torch.histogramdd(x_pred, grid_edges, density=True).hist
    grid_values_pred /= torch.sum(grid_values_pred)

    fig, axs = plt.subplots(ncols=2, figsize=(5.0, 2.75), sharex=True, sharey=True)
    axs[0].pcolormesh(grid_coords[0], grid_coords[1], grid_values_pred.T)
    axs[1].pcolormesh(grid_coords[0], grid_coords[1], grid_values_true.T)
    axs[0].set_title("PRED", fontsize="medium")
    axs[1].set_title("TRUE", fontsize="medium")
    return fig, axs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default="ring")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kind", type=str, default="reg")
    args = parser.parse_args()

    path = pathlib.Path(__file__)
    output_dir = os.path.join("outputs", path.stem)
    os.makedirs(output_dir, exist_ok=True)

    log_prob_func = get_log_prob_func(args.dist)

    ndim = 2
    nchains = args.chains
    ndraws = args.n // nchains
    theta_init = torch.randn((nchains, ndim))

    if args.kind == "reg":
        draws, accepts, depths = ment.samp.nurs.sample_nurs(
            log_prob_func=log_prob_func,
            theta_init=theta_init,
            n_draws=ndraws,
            step_size=0.2,
            max_doublings=10,
            threshold=1e-5,
        )
    elif args.kind == "ssa":
        draws, accepts, depths = ment.samp.nurs.sample_nurs_ssa(
            log_prob_func=log_prob_func,
            theta_init=theta_init,
            n_draws=ndraws,
            min_step_size=0.2,
            max_tree_doublings=10,
            max_step_doublings=8,
            threshold=1e-5,
        )

    draws = draws.reshape(draws.shape[0] * draws.shape[1], draws.shape[2])

    fig, axs = plot_samples(log_prob_func, draws)
    plt.savefig(os.path.join(output_dir, "fig_samp.png"))
    plt.show()
