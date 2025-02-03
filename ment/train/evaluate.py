import numpy as np
from ..core import MENT
from ..sim import simulate
from ..utils import unravel


class Evaluator:
    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples

    def __call__(self, model: MENT, x: np.ndarray = None) -> dict:
        transforms = model.transforms
        diagnostics = model.diagnostics
        projections = model.projections

        if x is None:
            x = model.unnormalize(model.sample(self.n_samples))

        x = x[: self.n_samples, :]

        simulate(x, transforms, diagnostics)

        discrepancy_vector = []
        for hist_meas, hist_pred in zip(unravel(projections), unravel(diagnostics)):
            y_meas = hist_meas.values
            y_pred = hist_pred.values
            discrepancy_vector.append(np.mean(np.abs(y_pred - y_meas)))
        discrepancy = sum(discrepancy_vector) / len(discrepancy_vector)

        result = {
            "mean_abs_error": discrepancy,
        }
        return result
