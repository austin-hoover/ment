import torch
import ment


def test_coords_to_edges_and_edges_to_coords_round_trip():
    coords = torch.tensor([0.5, 1.5, 2.5, 3.5])

    edges = ment.coords_to_edges(coords)
    recovered_coords = ment.edges_to_coords(edges)

    assert torch.allclose(edges, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(recovered_coords, coords)


def test_weighted_average():
    values = torch.tensor([1.0, 2.0, 3.0])
    weights = torch.tensor([1.0, 2.0, 1.0])

    result = ment.weighted_average(values, weights)

    assert torch.allclose(result, torch.tensor(2.0))


def test_rotation_matrix_rotates_points_clockwise_by_angle():
    matrix = ment.rotation_matrix(torch.pi / 2.0)
    point = torch.tensor([[1.0, 0.0]])

    rotated = torch.matmul(point, matrix.T)

    assert torch.allclose(rotated, torch.tensor([[0.0, -1.0]]), atol=1e-6)
