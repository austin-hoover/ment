import numpy as np
import scipy.optimize


class Profile:
    def __init__(self, coords: np.ndarray, values: np.ndarray) -> None:
        self.coords = np.array(coords)
        self.values = np.array(values)
        if np.any(self.values < 0.0):
            self.values = self.values - np.min(self.values)

    def mean(self, frac_thresh: float = None) -> None:
        values = np.copy(self.values)
        if frac_thresh is not None:
            thresh = frac_thresh * np.max(values)
            values[values <= thresh] = 0.0
        return np.average(self.coords, weights=values)

    def std(self) -> None:
        return np.sqrt(self.variance())

    def variance(self) -> None:
        return np.average(np.square(self.coords - self.mean()), weights=self.values)

    def center(self, mean: float = None) -> None:
        if mean is None:
            mean = self.mean()
        self.coords = self.coords - mean

    def interpolate(self, int_coords: np.ndarray) -> None:
        interp = scipy.interpolate.interp1d(
            self.coords,
            self.values,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        int_values = interp(int_coords)
        self.values = np.copy(int_values)
        self.coords = np.copy(int_coords)

    def clip(self, xmin: float, xmax: float) -> None:
        idx = np.logical_and(self.coords >= xmin, self.coords <= xmax)
        self.coords = self.coords[idx]
        self.values = self.values[idx]

    def threshold(self, thresh: float, kind: str = "abs") -> None:
        if kind == "frac":
            thresh = thresh * np.max(self.values)
        self.values[self.values < thresh] = 0.0

    def normalize(self) -> None:
        self.values = self.values - np.min(self.values)
        self.values = (
            self.values / np.sum(self.values) / (self.coords[1] - self.coords[0])
        )


def fit_norm_dist(coords: np.ndarray, values: np.ndarray):
    popt, pcov = scipy.optimize.curve_fit(normal_distribution_1d, coords, values)
    (sigma,) = popt
    return sigma


def fit_unif_dist(coords: np.ndarray, values: np.ndarray):
    popt, pcov = scipy.optimize.curve_fit(uniform_distribution_1d, coords, values)
    (sigma,) = popt
    return sigma


def norm_dist(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def unif_dist(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    xmax = 2.0 * sigma
    if type(x) is np.ndarray:
        idx = np.abs(x) <= xmax
        density = np.zeros(len(x))
        density[idx] = 2.0 * np.sqrt(xmax**2 - x[idx] ** 2) / (np.pi * xmax**2)
        return density
    else:
        if abs(x) > xmax:
            return 0.0
        density = 2.0 * np.sqrt(xmax**2 - x**2) / (np.pi * r**2)
        return density
