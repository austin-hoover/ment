import numpy as np


def coords_to_edges(coords: np.ndarray) -> np.ndarray:
    delta = coords[1] - coords[0]
    edges = np.zeros(len(coords) + 1)
    edges[:-1] = coords - 0.5 * delta
    edges[-1] = coords[-1] + delta
    return edges


def edges_to_coords(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def get_grid_points(grid_coords) -> np.ndarray:
    return np.stack([C.ravel() for C in np.meshgrid(*grid_coords, indexing="ij")], axis=-1)
