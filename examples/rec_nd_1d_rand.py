"""ND reconstruction from random 1D projections."""
import argparse
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment
from ment.train.plot import Plotter
from ment.train.plot import PlotDistCorner
from ment.train.plot import PlotProj1D

plt.style.use("./style.mplstyle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        type=str,
        default="gaussian-mixture",
        choices=["gaussian-mixture", "rings", "gaussian", "waterbag", "kv"],
    )
    parser.add_argument("--ndim", type=int, default=6)
    parser.add_argument("--nmeas", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=64)
    parser.add_argument("--xmax", type=float, default=3.5)
    parser.add_argument(
        "--mode", type=str, default="reverse", choices=["reverse", "forward"]
    )
    parser.add_argument(
        "--samp-method", type=str, default="mh", choices=["mh", "grid", "hmc", "nurs"]
    )
    parser.add_argument("--samp-chain-steps", type=int, default=1000)
    parser.add_argument("--samp-burnin", type=int, default=10)
    parser.add_argument("--samp-warm-start", type=int, default=1)
    parser.add_argument("--samp-grid-res", type=int, default=15)
    parser.add_argument("--samp-grid-noise", type=float, default=0.0)
    parser.add_argument("--samp-nurs-max-doublings", type=int, default=5)
    parser.add_argument("--nsamp", type=int, default=100_000)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--eval-size", type=int, default=100_000)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    ndim = 2
    seed = 0

    path = pathlib.Path(__file__)
    timestamp = time.strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", path.stem, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Source distribution
    # ----------------------------------------------------------------------------------

    ndim = args.ndim
    xmax = args.xmax
    seed = args.seed

    dist = ment.dist.get_dist(args.dist, ndim=ndim, seed=seed)
    x_true = dist.sample(args.eval_size)
    x_true = x_true.float()

    limits = args.ndim * [(-xmax, xmax)]

    # Forward model
    # ----------------------------------------------------------------------------------

    transforms = []
    for _ in range(args.nmeas):
        direction = torch.randn(ndim)
        direction = direction / torch.linalg.norm(direction)
        transform = ment.ProjectionTransform1D(direction=direction)
        transforms.append(transform)

    bin_edges = torch.linspace(-xmax, xmax, args.nbins + 1)

    diagnostics = []
    for _ in transforms:
        diagnostic = ment.Histogram1D(axis=0, edges=bin_edges)
        diagnostics.append([diagnostic])

    # Simulate data
    # ----------------------------------------------------------------------------------

    projections = ment.simulate_with_diag_update(
        x_true,
        transforms,
        diagnostics,
        thresh=5.00e-03,
    )

    # Reconstruction model
    # ----------------------------------------------------------------------------------

    prior = ment.GaussianPrior(ndim=ndim, scale=1.0)

    sampler = None

    if args.samp_method == "grid":
        sampler = ment.samp.GridSampler(
            limits=limits,
            shape=(ndim * [args.samp_grid_res]),
            noise=args.samp_grid_noise,
        )

    if args.samp_method in ["hmc", "nurs", "mh"]:
        chains = args.nsamp // args.samp_chain_steps
        start = 0.5 * torch.randn(chains, ndim)

    if args.samp_method == "mh":
        prop_cov = (0.5**2) * torch.eye(ndim)
        sampler = ment.MetropolisHastingsSampler(
            ndim=ndim,
            start=start,
            proposal_cov=prop_cov,
            burnin=args.samp_burnin,
            shuffle=True,
            verbose=1,
            noise=0.10,  # slight smoothing
            noise_type="gaussian",
            warm_start=args.samp_warm_start,
        )

    if args.samp_method == "hmc":
        sampler = ment.HamiltonianMonteCarloSampler(
            ndim=ndim,
            start=start,
            step_size=0.25,
            steps_per_samp=10,
            burnin=args.samp_burnin,
            verbose=1,
            warm_start=args.samp_warm_start,
        )

    if args.samp_method == "nurs":
        sampler = ment.NURSSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)),
            step_size=1,
            max_doublings=args.samp_nurs_max_doublings,
            threshold=1e-5,
            verbose=1,
            warm_start=args.samp_warm_start,
        )

    model = ment.MENT(
        ndim=ndim,
        transforms=transforms,
        projections=projections,
        prior=prior,
        sampler=sampler,
        nsamp=args.nsamp,
        mode="forward",
        verbose=True,
    )

    # Training
    # ----------------------------------------------------------------------------------

    plot_nsamp = x_true.shape[0]

    plot_model = Plotter(
        n_samples=plot_nsamp,
        plot_proj=[
            PlotProj1D(log=False),
        ],
        plot_dist=[
            PlotDistCorner(
                fig_kws=dict(figsize=(ndim * 1.4, ndim * 1.4)),
                limits=(ndim * [(-xmax, xmax)]),
                bins=64,
            ),
        ],
    )

    eval_model = ment.train.Evaluator(nsamp=plot_nsamp)

    trainer = ment.train.Trainer(
        model,
        plot_func=plot_model,
        eval_func=eval_model,
        output_dir=output_dir,
    )

    trainer.train(iters=args.iters, lr=args.lr)

    # Evaluation
    # ----------------------------------------------------------------------------------

    x_pred = model.unnormalize(model.sample(1_000_000))

    grid = ment.train.plot.CornerGrid(ndim, figsize=(ndim * 1.4, ndim * 1.4))
    for i, x in enumerate([x_true, x_pred]):
        color = ["black", "red"][i]
        grid.plot(
            x,
            limits=limits,
            bins=64,
            proc_kws=dict(scale="max", blur=1.0),
            kind="contour",
            colors=color,
            diag_kws=dict(color=color, kind="line"),
            levels=np.linspace(0.01, 1.0, 7),
        )
    plt.savefig(os.path.join(output_dir, "figures", "fig_corner_final"))
    plt.close("all")


if __name__ == "__main__":
    main(parse_args())
