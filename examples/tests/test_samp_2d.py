"""Test sampling algorithms (2D)."""
import argparse
import os
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import zuko

import ment

plt.style.use("../style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100_000)
parser.add_argument("--chains", type=int, default=100)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Create distribution
# --------------------------------------------------------------------------------------


class RingDistribution:
    def __init__(self) -> None:
        self.ndim = 2

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]
        log_prob = torch.sin(torch.pi * x1) - 2.0 * (x1**2 + x2**2 - 2.0) ** 2
        return torch.exp(log_prob)

    def prob_grid(
        self, shape: tuple[int], limits: list[tuple[float, float]]
    ) -> torch.Tensor:
        edges = [
            torch.linspace(limits[i][0], limits[i][1], shape[i] + 1)
            for i in range(self.ndim)
        ]
        coords = [0.5 * (e[:-1] + e[1:]) for e in edges]
        points = torch.stack(
            [c.ravel() for c in torch.meshgrid(*coords, indexing="ij")], axis=-1
        )
        values = self.prob(points)
        values = values.reshape(shape)
        return values, coords


ndim = 2
xmax = 3.0
dist = RingDistribution()

grid_limits = 2 * [(-xmax, xmax)]
grid_shape = (64, 64)
grid_values, grid_coords = dist.prob_grid(grid_shape, grid_limits)


# Run samplers
# --------------------------------------------------------------------------------------


def plot_samples(x_pred: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = plt.subplots(ncols=2, figsize=(5.0, 2.75), sharex=True, sharey=True)
    hist, edges = np.histogramdd(x_pred, bins=80, range=grid_limits)
    axs[0].pcolormesh(edges[0], edges[1], hist.T)
    axs[1].pcolormesh(grid_coords[0], grid_coords[1], grid_values.T)
    axs[0].set_title("PRED", fontsize="medium")
    axs[1].set_title("TRUE", fontsize="medium")
    return fig, axs


def make_sampler(name: str) -> ment.Sampler:
    sampler = None
    if name == "grid":
        sampler = ment.GridSampler(limits=grid_limits, shape=grid_shape)
    elif name == "mh":
        chains = args.chains
        start = torch.randn((chains, ndim)) * 0.25
        proposal_cov = torch.eye(ndim) * 0.1
        sampler = ment.MetropolisHastingsSampler(
            ndim=ndim,
            start=start,
            proposal_cov=proposal_cov,
            verbose=1,
        )
    elif name == "hmc":
        chains = args.chains
        step_size = 0.21
        steps_per_samp = 10
        sampler = ment.HamiltonianMonteCarloSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)) * 0.25**2,
            step_size=step_size,
            steps_per_samp=steps_per_samp,
            burnin=10,
            verbose=1,
        )
    elif name == "nurs":
        chains = args.chains
        sampler = ment.NURSSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)),
            step_size=0.3,
            max_doublings=10,
            threshold=1e-5,
        )
    elif name == "flow":
        flow = zuko.flows.NSF(features=ndim, transforms=3, hidden_features=[64] * 3)
        flow = zuko.flows.Flow(flow.transform.inv, flow.base)
        sampler = ment.FlowSampler(
            ndim=ndim,
            flow=flow,
            unnorm_matrix=None,
            train_kws=dict(
                iters=1000,
                batch_size=512,
            ),
        )
    elif name == "svgd":
        kernel = ment.samp.svgd.RBFKernel(sigma=0.2)
        sampler = ment.SVGDSampler(
            ndim=ndim, kernel=kernel, train_kws=dict(iters=500, lr=0.1), verbose=1
        )
    else:
        raise ValueError

    return sampler


def evaluate_sampler(name: str, size: int) -> None:
    sampler = make_sampler(name)

    if name == "svgd":
        size = min(size, 1000)

    x = sampler(dist.prob, size=size)

    fig, axs = plot_samples(x)
    plt.savefig(os.path.join(output_dir, f"fig_samp_{name}.png"))
    plt.close("all")

    pprint(sampler.results)


for name in ["grid", "mh", "hmc", "nurs", "flow", "svgd"]:
    print(name.upper())
    evaluate_sampler(name=name, size=args.n)
