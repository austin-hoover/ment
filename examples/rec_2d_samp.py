"""2D MENT reconstruction using particle sampling (forward mode)."""
import argparse
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment

plt.style.use("./style.mplstyle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        type=str,
        default="swissroll",
        choices=[
            "eight-gaussians",
            "galaxy",
            "gaussian-mixture",
            "hollow",
            "kv",
            "pinwheel",
            "rings",
            "swissroll",
            "two-spirals",
            "waterbag",
        ],
    )
    parser.add_argument("--nmeas", type=int, default=7)
    parser.add_argument("--nbins", type=int, default=80)
    parser.add_argument("--xmax", type=float, default=4.0)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.90)
    parser.add_argument(
        "--samp-method", type=str, default="mh", choices=["mh", "grid", "hmc", "nurs"]
    )
    parser.add_argument("--samp-chain-steps", type=int, default=1000)
    parser.add_argument("--samp-burnin", type=int, default=10)
    parser.add_argument("--samp-warm-start", type=int, default=1)
    parser.add_argument("--samp-grid-res", type=int, default=128)
    parser.add_argument("--samp-grid-noise", type=float, default=0.0)
    parser.add_argument("--samp-nurs-max-doublings", type=int, default=5)
    parser.add_argument("--nsamp", type=int, default=100_000)
    parser.add_argument("--show", action="store_true")
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

    dist = ment.dist.get_dist(args.dist, ndim=ndim, seed=seed, normalize=True)
    x_true = dist.sample(1_000_000)

    # Forward model
    # ----------------------------------------------------------------------------------

    transforms = []
    for i in range(args.nmeas):
        angle = torch.pi * (i / args.nmeas)
        matrix = ment.utils.rotation_matrix(angle)
        transform = ment.LinearTransform(matrix)
        transforms.append(transform)

    bin_edges = torch.linspace(-args.xmax, args.xmax, args.nbins)
    diagnostics = []
    for _ in transforms:
        diagnostic = ment.Histogram1D(axis=0, edges=bin_edges)
        diagnostics.append([diagnostic])

    # Simulate data
    # ----------------------------------------------------------------------------------

    projections = ment.simulate(x_true, transforms, diagnostics)

    # Reconstruction model
    # ----------------------------------------------------------------------------------

    prior = ment.GaussianPrior(ndim=2, scale=1.0)

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
        mode="forward",
        nsamp=args.nsamp,
    )

    # Training
    # ----------------------------------------------------------------------------------

    def eval_model(model: ment.MENT) -> None:
        results = {}

        x_pred = model.sample(1_000_000)
        projections_pred = ment.unravel(
            ment.simulate(x_pred, model.transforms, model.diagnostics)
        )
        projections_true = ment.unravel(model.projections)

        pred_error = 0.0
        for proj_pred, proj_true in zip(projections_pred, projections_true):
            pred_error += torch.mean(torch.abs(proj_pred.values - proj_true.values))
        pred_error = pred_error / len(projections_pred)
        results["prediction_error"] = float(pred_error)
        return results

    def plot_model(model: ment.MENT) -> list[plt.Figure]:
        figs = []

        # Sample particles
        x_pred = model.sample(1_000_000)

        # Simulate data
        projections_true = ment.unravel(model.projections)
        projections_pred = ment.unravel(
            ment.simulate(x_pred, model.transforms, model.diagnostics)
        )

        # Plot distribution
        limits = 2 * [(-args.xmax, args.xmax)]

        fig, axs = plt.subplots(ncols=2, figsize=(6.0, 3.0))
        for i, ax in enumerate(axs):
            grid_values, grid_edges = np.histogramdd(x_pred, bins=100, range=limits)
            ax.pcolormesh(grid_edges[0], grid_edges[1], grid_values.T)
        figs.append(fig)

        # Plot simulated vs. measured projections.
        ncols = min(args.nmeas, 7)
        nrows = int(np.ceil(args.nmeas / ncols))

        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(1.90 * ncols, 1.25 * nrows),
            sharex=True,
            sharey=True,
        )
        for index in range(len(projections_true)):
            ax = axs.flat[index]
            proj_true = projections_true[index]
            proj_pred = projections_pred[index]
            scale = proj_true.values.max()

            ax.plot(proj_true.coords, proj_true.values / scale, color="lightgray")
            ax.plot(
                proj_pred.coords,
                proj_pred.values / scale,
                color="black",
                marker=".",
                lw=0,
                ms=1.0,
            )
            ax.set_ylim(ax.get_ylim()[0], 1.25)
            ax.set_xlim(limits[0])
        figs.append(fig)

        return figs

    for iteration in range(args.iters):
        print("\nITERATION =", iteration)

        if iteration > 0:
            model.gauss_seidel_step(lr=args.lr)

        eval_results = eval_model(model)
        print(eval_results)

        figs = plot_model(model)
        for i, fig in enumerate(figs):
            filename = f"fig_{i:02.0f}_{iteration:03.0f}"
            filename = os.path.join(output_dir, filename)
            fig.savefig(filename)
            if args.show:
                plt.show()
        plt.close("all")

    print(output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
