import abc
import itertools
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Union

import numpy as np
import torch
from tqdm import tqdm


def unravel(iterable):
    return list(itertools.chain.from_iterable(iterable))


def coords_to_edges(coords: torch.Tensor) -> torch.Tensor:
    delta = coords[1] - coords[0]
    edges = torch.zeros(len(coords) + 1)
    edges[:-1] = coords - 0.5 * delta
    edges[-1] = coords[-1] + delta
    return edges


def edges_to_coords(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])


def get_grid_points(grid_coords: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [c.ravel() for c in torch.meshgrid(*grid_coords, indexing="ij")], axis=-1
    )


def wrap_tqdm(iterable, verbose=True):
    return tqdm(iterable) if verbose else iterable


def random_choice(
    items: torch.Tensor, size: int, pdf: torch.Tensor, rng: torch.Generator = None
) -> torch.Tensor:
    idx = torch.multinomial(pdf, num_samples=size, replacement=True, generator=rng)
    return items[idx]


def random_shuffle(items: torch.Tensor, rng: torch.Generator = None) -> torch.Tensor:
    idx = torch.randperm(items.shape[0])
    return items[idx]


def random_uniform(
    lb: torch.Tensor | float,
    ub: torch.Tensor | float,
    size: int,
    rng: torch.Generator = None,
    device: torch.device = None,
) -> torch.Tensor:
    return lb + (ub - lb) * torch.rand(size, device=device, generator=rng)


def rotation_matrix(angle: float) -> torch.Tensor:
    angle = torch.tensor(float(angle))
    matrix = torch.zeros((2, 2))
    matrix[0, 0] = +torch.cos(angle)
    matrix[0, 1] = +torch.sin(angle)
    matrix[1, 0] = -torch.sin(angle)
    matrix[1, 1] = +torch.cos(angle)
    return matrix


class Logger(abc.ABC):
    # https://github.com/deepmind/acme
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: Mapping[str, Any]) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""


class ListLogger(Logger):
    """Manually save the data to the class in a dict."""

    def __init__(self, path: str = None, freq: int = 1):
        self.path = path
        if self.path:
            if not pathlib.Path(self.path).parent.exists():
                pathlib.Path(self.path).parent.mkdir(exist_ok=True, parents=True)
        self.freq = freq
        self.history: Dict[str, List[Union[np.ndarray, float, int]]] = {}
        self.print_warning: bool = False
        self.iteration = 0

    def write(self, data: Mapping[str, Any]) -> None:
        for key, value in data.items():
            if key in self.history:
                try:
                    value = float(value)
                except:
                    pass
                self.history[key].append(value)
            else:
                if isinstance(value, np.ndarray):
                    assert np.size(value) == 1
                    value = float(value)
                else:
                    if isinstance(value, float) or isinstance(value, int):
                        pass
                    else:
                        if not self.print_warning:
                            print("non numeric history values being saved")
                            self.print_warning = True
                self.history[key] = [value]

        self.iteration += 1
        if self.path and ((self.iteration + 1) % self.freq == 0):
            pickle.dump(self.history, open(self.path, "wb"))  # overwrite

    def close(self) -> None:
        if self.path:
            pickle.dump(self.history, open(self.path, "wb"))
