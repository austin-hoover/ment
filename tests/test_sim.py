import torch
import ment


class _TwoRankComm:
    def Get_size(self):
        return 2

    def Get_rank(self):
        return 0

    def Allreduce(self, send, recv):
        # Pretend rank 1 contributed one particle to the second bin.
        recv[:] = send + torch.tensor([0.0, 1.0]).numpy()


class _FixedSampler:
    def __call__(self, prob_func, size, **kws):
        assert size == 1
        return torch.tensor([[0.25]])


def test_identity_transform_forward_and_inverse_return_input():
    transform = ment.IdentityTransform()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert torch.allclose(transform.forward(x), x)
    assert torch.allclose(transform.inverse(x), x)


def test_linear_transform_forward_and_inverse_round_trip():
    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    transform = ment.LinearTransform(matrix)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    z = transform.forward(x)
    recovered = transform.inverse(z)

    assert torch.allclose(z, torch.tensor([[2.0, 6.0], [6.0, 12.0]]))
    assert torch.allclose(recovered, x)


def test_linear_transform_to_moves_matrix_and_inverse():
    transform = ment.LinearTransform(torch.eye(2)).to("cpu")

    assert transform.device == torch.device("cpu")
    assert transform.matrix.device == transform.device
    assert transform.matrix_inv.device == transform.device


def test_composed_transform_forward_and_inverse_round_trip():
    scale = ment.LinearTransform(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    rotate = ment.LinearTransform(ment.rotation_matrix(torch.pi / 2.0))
    transform = ment.ComposedTransform(scale, rotate)

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    z = transform.forward(x)
    recovered = transform.inverse(z)

    assert torch.allclose(recovered, x, atol=1e-6)


def test_composed_transform_to_moves_child_transforms():
    transforms = (
        ment.IdentityTransform(),
        ment.LinearTransform(torch.eye(2)),
    )
    transform = ment.ComposedTransform(*transforms).to("cpu")

    assert transform.device == torch.device("cpu")
    assert all(child.device == transform.device for child in transforms)


def test_simulate_returns_copied_diagnostics_with_binned_values():
    x = torch.tensor([[0.25], [0.75], [1.25], [1.75]])
    transforms = [ment.IdentityTransform()]
    original_hist = ment.Histogram1D(axis=0, edges=torch.tensor([0.0, 1.0, 2.0]))
    diagnostics = [[original_hist]]

    result = ment.simulate(x, transforms, diagnostics)

    simulated_hist = result[0][0]

    assert simulated_hist is not original_hist
    assert torch.allclose(original_hist.values, torch.zeros(2))
    assert torch.allclose(
        torch.sum(simulated_hist.values * simulated_hist.bin_width), torch.tensor(1.0)
    )


def test_forward_simulation_partitions_samples_and_reduces_counts():
    projection = ment.Histogram1D(
        axis=0,
        edges=torch.tensor([0.0, 1.0, 2.0]),
        values=torch.ones(2),
    )
    model = ment.MENT(
        ndim=1,
        transforms=[ment.IdentityTransform()],
        projections=[[projection]],
        prior=ment.GaussianPrior(ndim=1, scale=1.0),
        sampler=_FixedSampler(),
        nsamp=2,
        mode="forward",
        verbose=0,
        comm=_TwoRankComm(),
    )

    simulated = model.simulate_single(index=0, diag_index=0)

    assert model.local_sample_size(3) == 2
    assert torch.allclose(simulated.values, torch.tensor([0.5, 0.5]))


def test_reverse_integration_returns_one_value_per_multidimensional_grid_point():
    edges = [torch.linspace(-2.0, 2.0, 5), torch.linspace(-3.0, 3.0, 7)]
    projection = ment.HistogramND(
        axis=(0, 2),
        edges=edges,
        values=torch.ones(4, 6),
    )
    model = ment.MENT(
        ndim=4,
        transforms=[ment.IdentityTransform()],
        projections=[[projection]],
        prior=ment.GaussianPrior(ndim=4, scale=1.0),
        sampler=ment.GridSampler(limits=4 * [(-3.0, 3.0)], shape=4 * (4,)),
        integration_limits=[[[(-2.0, 2.0), (-3.0, 3.0)]]],
        integration_size=4,
        mode="reverse",
        verbose=0,
    )

    simulated = model.simulate_single(index=0, diag_index=0)

    assert simulated.values.shape == projection.shape
    assert torch.isclose(
        torch.sum(simulated.values * simulated.bin_volume), torch.tensor(1.0)
    )
