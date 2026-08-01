import math
import torch
import ment


def test_infinite_uniform_prior_returns_ones():
    prior = ment.InfiniteUniformPrior(ndim=3)
    x = torch.zeros((5, 3))

    prob = prior.prob(x)

    assert torch.allclose(prob, torch.ones(5))


def test_gaussian_prior_probability_at_origin_for_unit_scale():
    prior = ment.GaussianPrior(ndim=2, scale=1.0)
    x = torch.tensor([[0.0, 0.0]])

    prob = prior.prob(x)

    expected = 1.0 / (2.0 * math.pi)
    assert torch.allclose(prob, torch.tensor([expected]), atol=1e-6)


def test_gaussian_prior_expands_scalar_scale_to_ndim():
    prior = ment.GaussianPrior(ndim=3, scale=2.0)

    assert prior.scale.shape == (3,)
    assert torch.allclose(prior.scale, torch.tensor([2.0, 2.0, 2.0]))
