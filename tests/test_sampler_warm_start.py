import torch

from ment.samp.hmc import HamiltonianMonteCarloSampler
from ment.samp.mh import MetropolisHastingsSampler
from ment.samp.nurs import NURSSampler


class FixedProposal:
    def sample(self, shape):
        return torch.ones((*shape, 2))


def test_mh_warm_start_reuses_chain_endpoints():
    initial = torch.zeros((2, 2))
    sampler = MetropolisHastingsSampler(
        ndim=2,
        start=initial,
        proposal_cov=torch.eye(2),
        warm_start=True,
    )
    sampler.proposal_dist = FixedProposal()

    sampler(lambda x: torch.ones(x.shape[0]), size=6)

    assert torch.equal(sampler.start, 2.0 * torch.ones_like(initial))


def test_mh_start_is_unchanged_by_default():
    initial = torch.zeros((2, 2))
    sampler = MetropolisHastingsSampler(
        ndim=2,
        start=initial,
        proposal_cov=torch.eye(2),
    )
    sampler.proposal_dist = FixedProposal()

    sampler(lambda x: torch.ones(x.shape[0]), size=6)

    assert torch.equal(sampler.start, initial)


def test_hmc_warm_start_reuses_final_state(monkeypatch):
    final_state = torch.tensor([[1.0, 2.0]])

    def sample_stub(*args, **kwargs):
        return {
            "samples": torch.zeros((3, 2)),
            "final_state": final_state,
            "acceptance_rate": torch.tensor(1.0),
        }

    monkeypatch.setattr("ment.samp.hmc.sample", sample_stub)
    sampler = HamiltonianMonteCarloSampler(
        ndim=2, start=torch.zeros((1, 2)), warm_start=True
    )

    sampler(lambda x: torch.ones(x.shape[0]), size=3)

    assert torch.equal(sampler.start, final_state)


def test_nurs_warm_start_reuses_chain_endpoints(monkeypatch):
    draws = torch.arange(12.0).reshape(3, 2, 2)

    def sample_stub(*args, **kwargs):
        return draws, None, None

    monkeypatch.setattr("ment.samp.nurs.sample_nurs", sample_stub)
    sampler = NURSSampler(ndim=2, start=torch.zeros((2, 2)), warm_start=True)

    sampler(lambda x: torch.ones(x.shape[0]), size=6)

    assert torch.equal(sampler.start, draws[-1])
