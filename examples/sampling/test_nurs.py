"""Test NURS sampler on 2D distribution."""

import argparse
import os
import pathlib
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import nurs


plt.style.use("../style.mplstyle")


def log_prob_func_ring(x: np.ndarray) -> float:
    x1 = x[0]
    x2 = x[1]
    return np.sin(np.pi * x1) - 2.0 * (x1**2 + x2**2 - 2.0) ** 2


def log_prob_func_normal(x: np.ndarray) -> float:
    return -0.5 * np.sum(x**2)


def get_log_prob_func(name: str) -> Callable:
    if name == "ring":
        return log_prob_func_ring
    if name == "normal":
        return log_prob_func_normal
    return None


def plot_samples(
    log_prob_func: Callable, x_pred: np.ndarray, xmax: float = 3.0, nbins: int = 64
) -> tuple:
    grid_limits = 2 * [(-xmax, xmax)]
    grid_shape = (nbins, nbins)
    grid_edges = [
        np.linspace(grid_limits[i][0], grid_limits[i][1], grid_shape[i] + 1)
        for i in range(len(grid_shape))
    ]
    grid_coords = [0.5 * (e[:-1] + e[1:]) for e in grid_edges]
    grid_points = np.stack(
        [c.ravel() for c in np.meshgrid(*grid_coords, indexing="ij")], axis=-1
    )

    grid_values_true = np.zeros(grid_points.shape[0])
    for i, x in enumerate(grid_points):
        grid_values_true[i] = np.exp(log_prob_func(x))
    grid_values_true = grid_values_true.reshape(grid_shape)
    grid_values_true /= np.sum(grid_values_true)

    grid_values_pred, _ = np.histogramdd(x_pred, grid_edges, density=True)
    grid_values_pred /= np.sum(grid_values_pred)

    fig, axs = plt.subplots(ncols=2, figsize=(5.0, 2.75), sharex=True, sharey=True)
    axs[0].pcolormesh(grid_coords[0], grid_coords[1], grid_values_pred.T)
    axs[1].pcolormesh(grid_coords[0], grid_coords[1], grid_values_true.T)
    axs[0].set_title("PRED", fontsize="medium")
    axs[1].set_title("TRUE", fontsize="medium")
    return fig, axs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default="ring")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    path = pathlib.Path(__file__)
    output_dir = os.path.join("outputs", path.stem)
    os.makedirs(output_dir, exist_ok=True)

    log_prob_func = get_log_prob_func(args.dist)

    ndim = 2
    rng = np.random.default_rng(args.seed)
    theta_init = np.zeros(ndim)
    draws, accepts, depths = nurs.sample_nurs(
        rng=rng,
        log_prob_func=log_prob_func,
        theta_init=theta_init,
        num_draws=args.n,
        step_size=0.2,
        max_doublings=10,
        threshold=1e-5,
    )

    fig, axs = plot_samples(log_prob_func, draws)
    plt.savefig(os.path.join(output_dir, "fig_samp.png"))
    plt.show()
