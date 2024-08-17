import numpy as np
from ..core import MENT
from ..sim import forward
from ..utils import unravel


class Evaluator:
    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples

    def __call__(self, model: MENT, X: np.ndarray = None) -> dict:
        transforms = model.transforms
        diagnostics = model.diagnostics
        projections = model.projections

        if X is None:
            X = model.unnormalize(model.sample(self.n_samples))

        X = X[: self.n_samples, :]

        projections_pred = forward(X, transforms, diagnostics)
        projections_meas = projections

        projections_pred = unravel(projections_pred)
        projections_meas = unravel(projections_meas)

        discrepancy_vector = []
        for y_pred, y_meas in zip(projections_pred, projections_meas):
            discrepancy_vector.append(np.mean(np.abs(y_pred - y_meas)))
        discrepancy = sum(discrepancy_vector) / len(discrepancy_vector)

        result = {
            "mean_abs_error": discrepancy,
        }
        return result
