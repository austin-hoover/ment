"""Interface for PyORBIT simulations."""

import torch

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode


def get_bunch_coords(bunch: Bunch, axis: tuple[int, ...] = None) -> torch.Tensor:
    if axis is None:
        axis = tuple(range(6))

    coords = torch.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        coords[i, 0] = bunch.x(i)
        coords[i, 1] = bunch.xp(i)
        coords[i, 2] = bunch.y(i)
        coords[i, 3] = bunch.yp(i)
        coords[i, 4] = bunch.z(i)
        coords[i, 5] = bunch.dE(i)
    return coords[:, axis]


def set_bunch_coords(bunch: Bunch, coords: torch.Tensor, axis: tuple[int, ...] = None) -> Bunch:
    if axis is None:
        axis = list(range(6))

    bunch.deleteAllParticles()
    bunch.compress()
    for i in range(coords.shape[0]):
        part_coords = torch.zeros(6)
        part_coords[axis] = coords[i]
        bunch.addParticle(*part_coords)
    return bunch


def reverse_bunch(bunch: Bunch) -> Bunch:
    size = bunch.getSize()
    for i in range(size):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
    return bunch


def track_bunch(
    bunch: Bunch,
    lattice: AccLattice,
    index_start: int = None,
    index_stop: int = None,
    **kws
) -> Bunch:
    if index_start is None:
        index_start = 0

    if index_stop is None:
        index_stop = len(lattice.getNodes()) - 1

    reverse = index_start > index_stop
    node_start = lattice.getNodes()[index_start]
    node_stop = lattice.getNodes()[index_stop]

    if reverse:
        bunch = reverse_bunch(bunch)
        lattice.reverseOrder()

    lattice.trackBunch(
        bunch,
        index_start=lattice.getNodeIndex(node_start),
        index_stop=lattice.getNodeIndex(node_stop),
        **kws,
    )

    if reverse:
        bunch = reverse_bunch(bunch)
        lattice.reverseOrder()

    return bunch


class ORBITTransform:
    """Uses ORBIT simulation to transform NumPy arrays."""

    def __init__(
        self,
        lattice: AccLattice,
        bunch: Bunch,
        axis: tuple[int, ...],
        index_start: int,
        index_stop: int,
    ) -> None:
        self.lattice = lattice
        self.bunch = bunch
        self.axis = axis
        self.index_start = index_start
        self.index_stop = index_stop

    def track_bunch(self) -> Bunch:
        bunch = Bunch()
        self.bunch.copyBunchTo(bunch)
        self.lattice.trackBunch(
            bunch,
            index_start=self.index_start,
            index_stop=self.index_stop,
        )
        return bunch

    def track_bunch_reverse(self) -> Bunch:
        bunch = Bunch()
        self.bunch.copyBunchTo(bunch)

        self.lattice.reverseOrder()
        bunch = reverse_bunch(bunch)
        self.lattice.trackBunch(
            bunch,
            index_start=self.index_stop,
            index_stop=self.index_start,
        )
        bunch = reverse_bunch(bunch)
        self.lattice.reverseOrder()

        return bunch

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.bunch = set_bunch_coords(self.bunch, x, axis=self.axis)
        bunch = self.track_bunch()
        return get_bunch_coords(bunch, axis=self.axis)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        self.bunch = set_bunch_coords(self.bunch, x, axis=self.axis)
        bunch = self.track_bunch_reverse()
        return get_bunch_coords(bunch, axis=self.axis)
