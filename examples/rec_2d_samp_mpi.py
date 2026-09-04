"""2D MENT reconstruction using particle sampling (forward mode) with MPI."""

import argparse
import math
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpi4py import MPI

import ment

plt.style.use(pathlib.Path(__file__).with_name("style.mplstyle"))


MCMC_METHODS = {"hmc", "mh", "nurs"}


def local_count(total: int, rank: int, size: int) -> int:
    """Return one rank's balanced share of a global count."""
    quotient, remainder = divmod(total, size)
    return quotient + int(rank < remainder)


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
        "--samp-method",
        type=str,
        default="grid",
        choices=["grid", "mh", "nurs", "hmc", "flow"],
    )
    parser.add_argument(
        "--samp-size",
        type=int,
        default=100_000,
        help="Fixed global projection sample count (divided among MPI ranks).",
    )
    parser.add_argument("--samp-grid-res", type=int, default=100)
    parser.add_argument("--samp-grid-noise", type=float, default=0.0)
    parser.add_argument(
        "--samp-chain-steps",
        type=int,
        default=1_000,
        help="Minimum number of retained draws per MCMC chain.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    ndim = 2
    seed = 0

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank() if mpi_comm is not None else 0
    mpi_size = mpi_comm.Get_size() if mpi_comm is not None else 1
    torch.manual_seed(seed + mpi_rank)

    local_chains = None
    sample_size = args.samp_size
    if args.samp_method in MCMC_METHODS:
        if args.samp_chain_steps < 1:
            raise ValueError("samp-chain-steps must be positive.")
        if sample_size < mpi_size * args.samp_chain_steps:
            raise ValueError(
                f"samp-size ({sample_size}) must be at least MPI ranks * "
                f"samp-chain-steps ({mpi_size * args.samp_chain_steps})."
            )

        local_samples = local_count(sample_size, mpi_rank, mpi_size)
        local_chains = local_samples // args.samp_chain_steps

    path = pathlib.Path(__file__)
    timestamp = time.strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", path.stem, timestamp)
    if mpi_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Source distribution
    # ----------------------------------------------------------------------------------

    projections = None
    if mpi_rank == 0:
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

    if mpi_rank == 0:
        projections = ment.simulate(x_true, transforms, diagnostics)
    if mpi_comm is not None:
        projections = mpi_comm.bcast(projections, root=0)

    # Reconstruction model
    # ----------------------------------------------------------------------------------

    prior = ment.GaussianPrior(ndim=2, scale=1.0)

    if args.samp_method == "grid":
        sampler = ment.samp.GridSampler(
            limits=(2 * [(-args.xmax, args.xmax)]),
            shape=(args.samp_grid_res, args.samp_grid_res),
            noise=args.samp_grid_noise,
            seed=seed + mpi_rank,
        )
    if args.samp_method == "hmc":
        sampler = ment.HamiltonianMonteCarloSampler(
            ndim=ndim,
            start=torch.randn((local_chains, ndim)) * 0.25**2,
            step_size=0.25,
            steps_per_samp=10,
            burnin=10,
            verbose=int(mpi_rank == 0),
            seed=seed + mpi_rank,
        )
    if args.samp_method == "mh":
        sampler = ment.MetropolisHastingsSampler(
            ndim=ndim,
            start=torch.randn((local_chains, ndim)) * 0.25**2,
            proposal_cov=torch.eye(ndim) * 0.25**2,
            burnin=10,
            verbose=int(mpi_rank == 0),
            seed=seed + mpi_rank,
        )
    if args.samp_method == "nurs":
        sampler = ment.NURSSampler(
            ndim=ndim,
            start=torch.randn((local_chains, ndim)),
            step_size=1,
            max_doublings=10,
            threshold=1e-5,
            seed=seed + mpi_rank,
        )
    if args.samp_method in MCMC_METHODS:
        print(
            f"MPI_RANK={mpi_rank} NSAMP={local_samples} NCHAINS={local_chains}",
            flush=True,
        )
    model = ment.MENT(
        ndim=ndim,
        transforms=transforms,
        projections=projections,
        prior=prior,
        sampler=sampler,
        mode="forward",
        nsamp=sample_size,
        mpi_comm=mpi_comm,
        verbose=int(mpi_rank == 0),
    )

    if mpi_rank == 0:
        print(f"MPI ranks = {mpi_size}")
        print(f"global samples = {sample_size}")
        print(f"samples on rank 0 = {model.local_sample_size(sample_size)}")
        if args.samp_method in MCMC_METHODS:
            samples_per_rank = [
                local_count(sample_size, rank, mpi_size) for rank in range(mpi_size)
            ]
            chains_per_rank = [
                samples // args.samp_chain_steps for samples in samples_per_rank
            ]
            steps_per_chain = [
                math.ceil(samples / chains)
                for samples, chains in zip(samples_per_rank, chains_per_rank)
            ]
            print(f"global MCMC chains = {sum(chains_per_rank)}")
            print(f"MCMC chains per rank = {chains_per_rank}")
            print(f"MCMC steps per chain by rank = {steps_per_chain}")

    # Training
    # ----------------------------------------------------------------------------------

    def sample_parallel(model: ment.MENT, size: int) -> torch.Tensor | None:
        local_size = model.local_sample_size(size)
        x_local = model.unnormalize(model.sample(local_size))
        if mpi_comm is None:
            return x_local

        parts = mpi_comm.gather(x_local.detach().cpu().numpy(), root=0)
        if mpi_rank == 0:
            return torch.from_numpy(np.concatenate(parts, axis=0))
        return None

    def eval_model(model: ment.MENT, x_pred: torch.Tensor) -> dict[str, float]:
        results = {}

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

    def plot_model(model: ment.MENT, x_pred: torch.Tensor) -> list[plt.Figure]:
        figs = []

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
        if mpi_rank == 0:
            print("\nITERATION =", iteration, flush=True)

        if iteration > 0:
            model.gauss_seidel_step(lr=args.lr)

        x_pred = sample_parallel(model, sample_size)
        if mpi_rank == 0:
            eval_results = eval_model(model, x_pred)
            print(eval_results, flush=True)

            figs = plot_model(model, x_pred)
            for i, fig in enumerate(figs):
                filename = f"fig_{i:02.0f}_{iteration:03.0f}"
                filename = os.path.join(output_dir, filename)
                fig.savefig(filename)
                if args.show:
                    plt.show()
            plt.close("all")

    if mpi_rank == 0:
        print(output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
