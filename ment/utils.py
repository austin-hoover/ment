"""https://github.com/lollcat/fab-torch/blob/master/fab/utils/logging.py"""

import abc
import itertools
import pathlib
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def rotation_matrix(angle: float) -> np.ndarray:
    M = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    M = np.array(M)
    return M


def wrap_tqdm(iterable, verbose=True):
    return tqdm(iterable) if verbose else iterable


def unravel(iterable):
    return list(itertools.chain.from_iterable(iterable))


class Logger(abc.ABC):
    # copied from Acme: https://github.com/deepmind/acme
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
