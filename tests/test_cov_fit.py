import math

import numpy as np
import pytest
import torch
from scipy.optimize import OptimizeResult

import ment


def make_projection(axis: int = 0) -> ment.Histogram1D:
    return ment.Histogram1D(
        axis=axis,
        edges=torch.tensor([-1.5, -0.5, 0.5, 1.5]),
        values=torch.tensor([1.0, 0.0, 1.0]),
    )


def make_fitter(
    fitter_type: type[ment.CovFitterBase] = ment.CholeskyCovFitter, **kwargs
):
    return fitter_type(
        ndim=2,
        transforms=[ment.IdentityTransform()],
        projections=[[make_projection()]],
        nsamp=128,
        verbose=0,
        **kwargs,
    )


def test_linear_measurements_use_exact_analytic_loss():
    fitter = make_fitter()

    assert fitter.loss_mode == "analytic"
    assert fitter.loss_function(np.array([1.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert fitter.loss_function(np.array([2.0, 1.0, 0.0])) == pytest.approx(3.0)


@pytest.mark.parametrize("fitter_type", [ment.CholeskyCovFitter, ment.LinearCovFitter])
def test_set_cov_round_trips_with_unnormalization(fitter_type):
    unnorm_matrix = torch.diag(torch.tensor([2.0, 3.0]))
    covariance = torch.tensor([[4.0, 1.2], [1.2, 9.0]])
    fitter = make_fitter(fitter_type, unnorm_matrix=unnorm_matrix)

    fitter.set_cov(covariance)

    torch.testing.assert_close(fitter.build_cov(), covariance)


def test_fixed_base_samples_make_sample_loss_deterministic():
    fitter = ment.CholeskyCovFitter(
        ndim=2,
        transforms=[lambda x: x],
        projections=[[make_projection()]],
        nsamp=128,
        verbose=0,
        resample=False,
        seed=1234,
    )

    first = fitter.loss_function(fitter.params)
    second = fitter.loss_function(fitter.params)

    assert fitter.loss_mode == "sample"
    assert second == first


def test_sample_applies_unnormalization_without_refactorization(monkeypatch):
    unnorm_matrix = torch.tensor([[2.0, 0.5], [0.0, 3.0]])
    fitter = make_fitter(unnorm_matrix=unnorm_matrix)
    monkeypatch.setattr(
        fitter, "_draw_standard_normal", lambda size, reuse: torch.eye(2)
    )

    samples = fitter.sample(size=2)

    torch.testing.assert_close(samples, unnorm_matrix.T)
