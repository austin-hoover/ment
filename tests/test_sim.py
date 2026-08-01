import torch
import ment


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


def test_composed_transform_forward_and_inverse_round_trip():
    scale = ment.LinearTransform(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    rotate = ment.LinearTransform(ment.rotation_matrix(torch.pi / 2.0))
    transform = ment.ComposedTransform(scale, rotate)

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    z = transform.forward(x)
    recovered = transform.inverse(z)

    assert torch.allclose(recovered, x, atol=1e-6)


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
