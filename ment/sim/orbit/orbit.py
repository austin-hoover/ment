"""Interface for PyORBIT simulations."""
import time
import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.utils import speed_of_light

from .bunch import set_bunch_coords
from .bunch import get_bunch_coords
from .bunch import reverse_bunch


def track_bunch(bunch: Bunch, lattice: AccLattice, index_start: int = None, index_stop: int = None, **kws) -> Bunch:
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
        **kws
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

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.bunch = set_bunch_coords(self.bunch, X, axis=self.axis)
        bunch = self.track_bunch()
        X_out = get_bunch_coords(bunch, axis=self.axis)
        return X_out

    def inverse(self, X: np.ndarray) -> np.ndarray:
        self.bunch = set_bunch_coords(self.bunch, X, axis=self.axis)
        bunch = self.track_bunch_reverse()
        X_out = get_bunch_coords(bunch, axis=self.axis)
        return X_out