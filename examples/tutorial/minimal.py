"""Minimal 2D MENT example.

After installing MENT (``pip install -e .``), run from the repo root:
    python examples/minimal.py --mode reverse
    python examples/minimal.py --mode forward

Reverse mode integrates the density over unobserved coordinates to simulate
each projection. It needs inverse transforms and is useful in low dimensions.
This implementation assumes volume-preserving transforms.

Forward mode samples particles, transforms them, and bins them at the diagnostics.
It has sampling noise but avoids inverse transform, making it easier to work
with particle tracking codes. For high-dimensional problems, replace GridSampler
with an MCMC sampler.
"""

import argparse

import matplotlib.pyplot as plt
import torch

import ment


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(0)

    # 1. Generate an unknown distribution: two Gaussian blobs in 2D.
    centers = torch.tensor([[-1.0, -0.6], [1.0, 0.6]])
    x_true = centers[torch.randint(2, (50_000,))] + 0.4 * torch.randn(50_000, 2)

    # 2. A measurement is a transform followed by a diagnostic.
    # Rotating then histogramming axis 0 gives a projection at that angle.
    transforms = [
        ment.LinearTransform(ment.rotation_matrix(i * torch.pi / 5))
        for i in range(args.nmeas)
    ]
    edges = torch.linspace(-3.0, 3.0, 61)  # 60 bins
    diagnostics = [[ment.Histogram1D(axis=0, edges=edges)] for _ in transforms]

    # Lists are nested as [transform][diagnostic]: each transform may have
    # multiple diagnostics. Real measurements would replace these histograms.
    projections = ment.simulate(x_true, transforms, diagnostics)

    # 3. MENT receives only the projections, not the original particles.
    # It represents p(x) as a prior times one Lagrange multiplier function
    # (h function) per measurement.
    model = ment.MENT(
        ndim=2,
        transforms=transforms,
        projections=projections,
        prior=ment.GaussianPrior(ndim=2, scale=1.0),
        sampler=ment.GridSampler(
            limits=[(-3.0, 3.0), (-3.0, 3.0)], shape=(80, 80), seed=0
        ),
        mode=args.mode,
        nsamp=50_000,  # Forward only: particles per predicted projection.
        # Reverse only: integration bounds per [transform][diagnostic],
        # and total quadrature points over the unobserved coordinates.
        integration_limits=[[(-3.0, 3.0)] for _ in transforms],
        integration_size=100,
        integration_loop=False,
        verbose=0,
    )

    # 4. Each iteration adjusts the factors to match the measured projections
    # while maximizing entropy relative to the Gaussian prior.
    for iteration in range(5):
        # Update the model.
        if iteration > 0:
            model.gauss_seidel_step(lr=0.9)

        # Print the fit error.
        projections_pred = ment.unravel(model.simulate())
        projections_meas = ment.unravel(model.projections)
        errors = [
            torch.mean(torch.abs(pred.values - meas.values))
            for pred, meas in zip(projections_pred, projections_meas)
        ]
        error = float(torch.mean(torch.stack(errors)))
        print(f"iter={iteration} error={error:.4f}")

    # 5. Sample particles from the fitted density.
    x_reconstructed = model.sample(100_000)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, x, title in zip(
        axs[:2], [x_true, x_reconstructed], ["Source", "Reconstruction"]
    ):
        ax.hist2d(x[:, 0], x[:, 1], bins=60, range=[(-3, 3), (-3, 3)])
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")

    axs[2].plot(
        projections_meas[0].coords,
        projections_meas[0].values,
        color="black",
        label="Measured",
    )
    axs[2].plot(
        projections_pred[0].coords,
        projections_pred[0].values,
        "--",
        color="red",
        label="MENT",
    )
    axs[2].set(title="First projection", xlabel="x", ylabel="Density")
    axs[2].legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmeas", type=int, default=5)
    parser.add_argument("--mode", choices=["reverse", "forward"], default="reverse")
    main(parser.parse_args())
