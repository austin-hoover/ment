import torch

import ment


def make_cached_grid_model():
    edges = torch.linspace(-3.0, 3.0, 17)
    transforms = [
        ment.IdentityTransform(),
        ment.LinearTransform(ment.rotation_matrix(0.4)),
    ]
    diagnostics = [[ment.Histogram1D(axis=0, edges=edges)] for _ in transforms]

    generator = torch.Generator().manual_seed(123)
    x_true = torch.randn((20_000, 2), generator=generator)
    projections = ment.simulate(x_true, transforms, diagnostics)

    sampler = ment.GridSampler(
        limits=2 * [(-3.0, 3.0)],
        shape=2 * (32,),
        seed=456,
    )
    return ment.MENT(
        ndim=2,
        transforms=transforms,
        projections=projections,
        prior=ment.GaussianPrior(ndim=2, scale=1.0),
        sampler=sampler,
        mode="forward",
        nsamp=2_000,
        verbose=0,
    )


def test_cached_grid_sampling_returns_requested_shape():
    model = make_cached_grid_model()

    samples = model.sample(128)

    assert samples.shape == (128, 2)
    assert not hasattr(model.sampler, "grid_cache")
    assert model.grid_cache is not None
    assert model.grid_cache["prob_values"].shape == (32 * 32,)


def test_cached_grid_lagrange_values_refresh_after_gauss_seidel_update():
    model = make_cached_grid_model()
    model.sample(128)

    before = model.grid_cache["interp_values"][0][0].clone()
    model.gauss_seidel_step(lr=0.5)
    after = model.grid_cache["interp_values"][0][0]

    assert not torch.allclose(before, after)
    assert model.grid_cache["prob_values"].shape == (32 * 32,)
    assert torch.allclose(
        model._ensure_grid_cache().prob(model.grid_cache["points"]),
        model.grid_cache["prob_values"],
    )


def make_reverse_grid_model(cache_grid):
    sampler = ment.GridSampler(
        limits=2 * [(-2.5, 2.5)],
        shape=2 * (5,),
    )
    projection = ment.Histogram1D(
        axis=0,
        edges=sampler.edges[0],
        values=torch.ones(5),
    )
    integration_limits = [[(sampler.coords[1][0], sampler.coords[1][-1])]]
    return ment.MENT(
        ndim=2,
        transforms=[ment.IdentityTransform()],
        projections=[[projection]],
        prior=ment.GaussianPrior(ndim=2, scale=1.0),
        sampler=sampler,
        integration_limits=integration_limits,
        integration_size=5,
        mode="reverse",
        cache_grid=cache_grid,
        verbose=0,
    )


def test_reverse_integration_interpolates_cached_probability_grid():
    for integration_loop in (True, False):
        cached_model = make_reverse_grid_model(cache_grid=True)
        exact_model = make_reverse_grid_model(cache_grid=False)
        cached_model.integration_loop = integration_loop
        exact_model.integration_loop = integration_loop

        cached = cached_model.simulate_single(0, 0)
        exact = exact_model.simulate_single(0, 0)

        assert cached_model.grid_cache is not None
        assert cached_model.grid_cache["prob_values"].shape == (5 * 5,)
        assert torch.allclose(cached.values, exact.values)


def test_reverse_integration_does_not_enable_grid_cache_implicitly():
    model = make_reverse_grid_model(cache_grid=None)

    model.simulate_single(0, 0)

    assert model.grid_cache is None
