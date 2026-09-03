"""4D reconstruction from 2D projections (random x-y phase aadvances)."""
import argparse
import math
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment
from ment.train.plot import Plotter
from ment.train.plot import PlotDistCorner
from ment.train.plot import PlotProj2DContour

plt.style.use("./style.mplstyle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        type=str,
        default="gaussian-mixture",
        choices=["gaussian-mixture", "rings", "gaussian", "waterbag", "kv"],
    )
    parser.add_argument("--nmeas", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=64)
    parser.add_argument("--xmax", type=float, default=3.5)
    parser.add_argument("--mode", type=str, default="forward")
    parser.add_argument(
        "--samp-method",
        type=str,
        default="grid",
        choices=["grid", "mh", "nurs", "hmc", "flow"],
    )
    parser.add_argument("--samp-grid-res", type=int, default=32)
    parser.add_argument("--samp-grid-noise", type=float, default=0.0)
    parser.add_argument("--samp-chains", type=int, default=100)
    parser.add_argument("--samp-size", type=int, default=100_000)
    parser.add_argument("--int-size", type=int, default=(50**2))
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--eval-size", type=int, default=500_000)
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

    ndim = 4
    xmax = args.xmax
    seed = args.seed

    dist = ment.dist.get_dist(args.dist, ndim=ndim, seed=seed)
    x_true = dist.sample(args.eval_size)
    x_true = x_true.float()

    limits = ndim * [(-xmax, xmax)]

    # Forward model
    # ----------------------------------------------------------------------------------

    axis_meas = (0, 2)
    nmeas = args.nmeas

    phase_advances = ment.utils.random_uniform(0.0, math.pi, size=(args.nmeas, 2))
    transfer_matrices = []
    for mux, muy in phase_advances:
        matrix = torch.eye(ndim)
        matrix[0:2, 0:2] = ment.utils.rotation_matrix(mux)
        matrix[2:4, 2:4] = ment.utils.rotation_matrix(muy)
        transfer_matrices.append(matrix)

    transforms = []
    for matrix in transfer_matrices:
        transform = ment.LinearTransform(matrix)
        transforms.append(transform)

    bin_edges = 2 * [torch.linspace(-xmax, xmax, args.nbins + 1)]

    diagnostics = []
    for _ in transforms:
        diagnostic = ment.diag.HistogramND(axis=axis_meas, edges=bin_edges)
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

    if args.samp_method == "grid":
        sampler = ment.samp.GridSampler(
            limits=limits,
            shape=(ndim * [args.samp_grid_res]),
            noise=args.samp_grid_noise,
        )
    if args.samp_method == "hmc":
        chains = args.samp_chains
        sampler = ment.HamiltonianMonteCarloSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)) * 0.25**2,
            step_size=0.25,
            steps_per_samp=10,
            burnin=10,
            verbose=1,
        )
    if args.samp_method == "mh":
        chains = args.samp_chains
        sampler = ment.MetropolisHastingsSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)) * 0.25**2,
            proposal_cov=torch.eye(ndim) * 0.25**2,
            burnin=10,
            verbose=1,
        )
    if args.samp_method == "nurs":
        chains = args.samp_chains
        sampler = ment.NURSSampler(
            ndim=ndim,
            start=torch.randn((chains, ndim)),
            step_size=1,
            max_doublings=5,
            threshold=1e-5,
        )

    integration_limits = [limits[axis] for axis in range(ndim) if axis not in axis_meas]
    integration_limits = [[integration_limits]] * len(transforms)

    model = ment.MENT(
        ndim=ndim,
        transforms=transforms,
        projections=projections,
        prior=prior,
        sampler=sampler,
        integration_limits=integration_limits,
        integration_size=(args.int_size**2),
        nsamp=args.samp_size,
        mode=args.mode,
        verbose=True,
    )

    # Training
    # ----------------------------------------------------------------------------------

    plot_nsamp = x_true.shape[0]

    def plot_dist(x_pred: np.ndarray):
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
                linewidths=1.1,
            )
        return (grid.fig, grid.axs)

    plot_model = Plotter(
        n_samples=plot_nsamp,
        plot_proj=[PlotProj2DContour()],
        plot_dist=[plot_dist],
    )

    eval_model = ment.train.Evaluator(nsamp=100_000)

    trainer = ment.train.Trainer(
        model,
        plot_func=plot_model,
        eval_func=eval_model,
        output_dir=output_dir,
    )

    trainer.train(iters=args.iters, lr=args.lr)


if __name__ == "__main__":
    main(parse_args())
