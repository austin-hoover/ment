import numpy as np
from ..core import MENT
from ..sim import simulate
from ..utils import unravel


class Evaluator:
    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples

    def __call__(self, model: MENT, x: np.ndarray = None) -> dict:
        if x is None:
            x = model.unnormalize(model.sample(self.n_samples))
        x = x[:self.n_samples]

        projections_pred = simulate(x, model.transforms, model.diagnostics)
        projections_meas = model.projections

        discrepancy_vector = []
        for hist_meas, hist_pred in zip(unravel(projections_meas), unravel(projections_pred)):
            y_meas = hist_meas.values
            y_pred = hist_pred.values            
            discrepancy_vector.append(np.mean(np.abs(y_pred - y_meas)))
        discrepancy = sum(discrepancy_vector) / len(discrepancy_vector)

        result = {}
        result["mean_abs_error"] = discrepancy
        return result
