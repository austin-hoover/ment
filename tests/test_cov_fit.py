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


def test_fit_returns_covariance_at_optimizer_result(monkeypatch):
    fitter = make_fitter()
    optimum = np.array([2.0, 3.0, 0.5])
    optimizer_kwargs = {}

    def fake_minimize(function, initial, **_kwargs):
        optimizer_kwargs.update(_kwargs)
        function(initial)
        return OptimizeResult(x=optimum, fun=0.0, success=True)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    covariance, result = fitter.fit(method="simplex", iters=1)

    np.testing.assert_allclose(fitter.params, optimum)
    np.testing.assert_allclose(result.x, optimum)
    initial_simplex = optimizer_kwargs["options"]["initial_simplex"]
    assert initial_simplex[-1, -1] != 0.0
    torch.testing.assert_close(covariance, torch.tensor([[4.0, 1.0], [1.0, 9.25]]))


def test_least_squares_uses_a_float32_appropriate_difference_step():
    projection = make_projection()
    fitter = ment.CholeskyCovFitter(
        ndim=1,
        transforms=[ment.IdentityTransform()],
        projections=[[projection]],
        nsamp=32,
        bound=5.0,
        verbose=0,
    )
    fitter.params[:] = 2.0

    covariance, _ = fitter.fit(method="least_squares", iters=50)

    torch.testing.assert_close(covariance, torch.ones(1, 1), atol=1.0e-4, rtol=0.0)


def test_nelder_mead_uses_a_scale_aware_initial_simplex():
    target_covariance = torch.tensor([[2.0, -0.5], [-0.5, 1.0]])
    transforms = [
        ment.LinearTransform(ment.utils.rotation_matrix(angle))
        for angle in (0.0, math.pi / 2.0, math.pi / 4.0)
    ]
    target_variances = [2.0, 1.0, 1.0]
    projections = []
    for variance in target_variances:
        outer_weight = variance / 8.0
        projections.append(
            [
                ment.Histogram1D(
                    axis=0,
                    edges=torch.tensor([-3.0, -1.0, 1.0, 3.0]),
                    values=torch.tensor(
                        [outer_weight, 1.0 - variance / 4.0, outer_weight]
                    ),
                )
            ]
        )
    fitter = ment.CholeskyCovFitter(
        ndim=2,
        transforms=transforms,
        projections=projections,
        nsamp=32,
        bound=10.0,
        verbose=0,
    )

    covariance, result = fitter.fit(method="nelder-mead", iters=500)

    assert result.success
    torch.testing.assert_close(covariance, target_covariance, atol=2.0e-4, rtol=0.0)


def test_unknown_optimizer_reports_supported_methods():
    fitter = make_fitter()

    with pytest.raises(ValueError, match="Unknown optimization method"):
        fitter.fit(method="not-an-optimizer")
