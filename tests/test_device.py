import torch

import ment


def test_ment_device_moves_model_state_and_created_tensors():
    edges = torch.linspace(-2.0, 2.0, 9)
    projection = ment.Histogram1D(
        axis=0,
        edges=edges,
        values=torch.ones(edges.numel() - 1),
    )
    transform = ment.IdentityTransform()
    model = ment.MENT(
        ndim=2,
        transforms=[transform],
        projections=[[projection]],
        prior=ment.GaussianPrior(ndim=2, scale=1.0),
        sampler=ment.samp.GridSampler(limits=[(-2.0, 2.0), (-2.0, 2.0)], shape=(8, 8)),
        integration_limits=[[(-2.0, 2.0)]],
        integration_size=8,
        mode="reverse",
        device="cpu",
    )

    assert model.device == torch.device("cpu")
    assert transform.device == model.device
    assert model.unnorm_matrix.device == model.device
    assert model.projections[0][0].values.device == model.device
    assert model.projections[0][0].edges.device == model.device
    assert model.lagrange_functions[0][0].values.device == model.device
    assert model.prior.scale.device == model.device
    assert model.sampler.device == model.device
    assert model._get_integration_points(0, 0).device == model.device


def test_histogram_to_moves_all_tensor_state():
    histogram = ment.Histogram1D(
        axis=0,
        edges=torch.linspace(-1.0, 1.0, 5),
        direction=torch.tensor([1.0, 0.0]),
    ).to("cpu")

    assert histogram.values.device.type == "cpu"
    assert histogram.coords.device.type == "cpu"
    assert histogram.edges.device.type == "cpu"
    assert histogram.bin_size.device.type == "cpu"
    assert histogram.direction.device.type == "cpu"
