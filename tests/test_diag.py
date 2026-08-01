import pytest
import torch
import ment


def test_histogram1d_builds_coords_from_edges():
    edges = torch.tensor([0.0, 1.0, 2.0, 3.0])

    hist = ment.Histogram1D(axis=0, edges=edges)

    assert torch.allclose(hist.coords, torch.tensor([0.5, 1.5, 2.5]))
    assert hist.shape == 3
    assert torch.allclose(hist.values, torch.zeros(3))


def test_histogram1d_bin_normalizes_density():
    edges = torch.tensor([0.0, 1.0, 2.0, 3.0])
    hist = ment.Histogram1D(axis=0, edges=edges)
    x = torch.tensor([[0.25], [0.75], [1.25], [2.25]])

    values = hist.bin(x)

    assert values.shape == (3,)
    assert torch.allclose(torch.sum(values * hist.bin_width), torch.tensor(1.0))


def test_histogram1d_direction_projection():
    hist = ment.Histogram1D(
        edges=torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
        direction=torch.tensor([1.0, 0.0]),
    )
    x = torch.tensor([[1.5, 100.0], [-0.5, 100.0]])

    projected = hist.project(x)

    assert torch.allclose(projected, torch.tensor([1.5, -0.5]))


def test_histogram1d_var_raises_for_empty_histogram():
    hist = ment.Histogram1D(edges=torch.tensor([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="Histogram values are zero"):
        hist.var()


def test_histogramnd_builds_shape_and_grid_points():
    hist = ment.HistogramND(
        axis=(0, 1),
        edges=[
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([-1.0, 0.0, 1.0]),
        ],
    )

    grid_points = hist.get_grid_points()

    assert hist.shape == (2, 2)
    assert grid_points.shape == (4, 2)
