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
from ment.train.plot import CornerGrid

plt.style.use("../style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--ndim", type=int, default=6)
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


class GaussianMixtureDistribution:
    def __init__(self, locs: torch.Tensor, covs: torch.Tensor) -> None:
        self.dists = []
        for loc, cov in zip(locs, covs):
            dist = torch.distributions.MultivariateNormal(loc, cov)
            self.dists.append(dist)

        self.ndim = len(locs[0])
        self.nmodes = len(self.dists)

    def sample(self, size: int) -> torch.Tensor:
        sizes = torch.ones(self.nmodes) * (size // self.nmodes)

        indices = torch.arange(self.nmodes)
        if self.nmodes > 1:
            indices = indices[sizes > 0]

        x = torch.empty(0, device=sizes.device)
        for i in indices:
            dist = self.dists[i]
            size = int(sizes[i])
            x_k = dist.sample((size,))
            x = torch.cat((x, x_k), dim=0)
        return x

    def prob(self, x: torch.Tensor) -> None:
        p = torch.zeros(x.shape[0])
        for dist in self.dists:
            p += torch.exp(dist.log_prob(x))
        return p


ndim = args.ndim
nmodes = 7
seed = 11
xmax = 7.0

torch.manual_seed(seed)

dist_locs = []
dist_covs = []
for _ in range(nmodes):
    loc = 5.0 * (torch.rand(size=(ndim,)) - 0.5)
    std = 1.0 * (torch.rand(size=(ndim,))) + 0.5
    cov = torch.eye(ndim) * std**2
    dist_locs.append(loc)
    dist_covs.append(cov)

dist = GaussianMixtureDistribution(locs=dist_locs, covs=dist_covs)
x_true = dist.sample(args.n)


# Run samplers
# --------------------------------------------------------------------------------------


def make_sampler(name: str) -> ment.Sampler:
    chains = args.chains

    sampler = None
    if name == "mh":
        start = torch.randn((chains, ndim)) * 0.25
        proposal_cov = torch.eye(ndim) * 0.1
        sampler = ment.MetropolisHastingsSampler(
            ndim=ndim,
            start=start,
            proposal_cov=proposal_cov,
            verbose=2,
        )
    elif name == "hmc":
        sampler = ment.HamiltonianMonteCarloSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)) * 0.25**2,
            step_size=0.7,
            steps_per_samp=10,
            burnin=10,
            verbose=1,
        )
    elif name == "nurs":
        chains = args.chains
        sampler = ment.NURSSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)),
            step_size=0.7,
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
            verbose=2,
        )
    elif name == "svgd":
        kernel = ment.samp.svgd.RBFKernel(sigma=0.2)
        sampler = ment.SVGDSampler(
            ndim=ndim, kernel=kernel, train_kws=dict(iters=500, lr=0.1), verbose=2
        )
    return sampler


def plot_samples_2d(x_pred: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = plt.subplots(ncols=2, figsize=(5.0, 2.75), sharex=True, sharey=True)
    for ax, x in zip(axs, [x_pred, x_true]):
        ax.hist2d(x[:, 0], x[:, 1], bins=80, range=[(-xmax, xmax), (-xmax, xmax)])
    axs[0].set_title("PRED", fontsize="medium")
    axs[1].set_title("TRUE", fontsize="medium")
    return fig, axs


def plot_samples_corner(x: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    grid = CornerGrid(
        ndim=args.ndim,
        figsize=(ndim * 1.4, ndim * 1.4),
    )
    grid.plot(
        x,
        bins=85,
        limits=[(-xmax, xmax)] * args.ndim,
    )
    return (grid.fig, grid.axs)


def plot_samples_corner_overlay(x_pred: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    grid = CornerGrid(
        ndim=args.ndim,
        figsize=(ndim * 1.4, ndim * 1.4),
    )
    for i, x in enumerate([x_true, x_pred]):
        color = ["black", "red"][i]
        grid.plot(
            x,
            bins=64,
            limits=[(-xmax, xmax)] * args.ndim,
            kind="contour",
            proc_kws=dict(scale="max", blur=1.0),
            diag_kws=dict(color=color),
            levels=np.linspace(0.01, 1.0, 7),
            colors=color,
        )
    return (grid.fig, grid.axs)


def plot_samples(x_pred: torch.Tensor) -> None:
    fig, axs = plot_samples_2d(x_pred)
    plt.savefig(os.path.join(output_dir, f"fig_01_{name}.png"))
    plt.close("all")

    fig, axs = plot_samples_corner(x_pred)
    plt.savefig(os.path.join(output_dir, f"fig_02_{name}.png"))
    plt.close("all")

    fig, axs = plot_samples_corner(x_true)
    plt.savefig(os.path.join(output_dir, f"fig_02_true.png"))
    plt.close("all")

    fig, axs = plot_samples_corner_overlay(x_pred)
    plt.savefig(os.path.join(output_dir, f"fig_03_{name}.png"))
    plt.close("all")


def evaluate_sampler(name: str, size: int) -> None:
    sampler = make_sampler(name)

    if name == "svgd":
        size = min(size, 2000)

    x = sampler(dist.prob, size=size)
    plot_samples(x)
    pprint(sampler.results)


for name in ["mh", "hmc", "nurs", "flow", "svgd"]:
    print(name.upper())
    evaluate_sampler(name=name, size=args.n)
