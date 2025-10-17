import numpy as np
from ..core import MENT
from ..sim import simulate
from ..utils import unravel


def evaluate_model(model: MENT, nsamp: int) -> dict:
    x = model.unnormalize(model.sample(nsamp))
    projections_pred = unravel(simulate(x, model.transforms, model.diagnostics))
    projections_meas = unravel(model.projections)

    discrepancy_vector = []
    for projection_meas, projection_pred in zip(projections_meas, projections_pred):
        y_meas = projection_meas.values
        y_pred = projection_pred.values
        discrepancy = np.mean(np.abs(y_pred - y_meas))
        discrepancy_vector.append(discrepancy)

    discrepancy_vector = np.array(discrepancy_vector)
    discrepancy = np.sum(discrepancy_vector) / len(discrepancy_vector)

    cov_matrix = np.cov(x.T)

    result = {}
    result["discrepancy"] = discrepancy
    result["cov_matrix"] = cov_matrix
    return result


class Evaluator:
    def __init__(self, nsamp: int) -> None:
        self.nsamp = nsamp

    def __call__(self, model: MENT) -> dict:
        return evaluate_model(model, self.nsamp)
