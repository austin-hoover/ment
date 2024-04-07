import typing
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import torch
import torch.nn as nn
import zuko

from mentflow.entropy import EntropyEstimator
from mentflow.entropy import get_entropy_estimator
from mentflow.gen import GenModel
from mentflow.sim import Simulator
from mentflow.loss import get_loss_function
from mentflow.types_ import Model
from mentflow.types_ import PriorDistribution
from mentflow.utils import unravel


class MENTFlow(Model, nn.Module):
    """Flow-based maximum entropy tomography solver."""
    def __init__(
        self,
        gen: GenModel,
        prior: Type[nn.Module],
        entropy_estimator: str,
        transforms: List[Type[nn.Module]],
        diagnostics: List[List[Type[nn.Module]]],
        measurements: List[List[torch.Tensor]],
        discrepancy_function: str = "kld",
        penalty_parameter: float = 10.0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        gen : GenModel
            A trainable model that generates samples and (possibly) evalutes the probability 
            density.
        entropy_estimator : str
            Method to estimate entropy from samples and probability density. 
            Options: 
                - "mc": monte carlo
                - "cov": covariance matrix
                - "knn": k nearest neighbors
                - None: return 0
        prior : Type[nn.Module]
            A prior distribution for relative entropy estimates. If None, use absolute entropy.
        transforms : list[nn.Module]
            The ith transform is applied before the ith measurement.
        diagnostics : list[list[torch.nn.Module]
            diagnostics[i] is a list of diagnostics to apply after transform i.
        measurements : list[list[tensor]
            The measurement data for each diagnostic.
        discrepancy_function : {"kld", "mae", "mse"}
            Function used to estimate discrepancy between simulated and measured projections.
            - "kld": KL divergence
            - "mae": mean absolute error
            - "mse": mean squared error
        penalty_parameter : float
            Loss = H + penalty_parameter * |D|.
        """
        super().__init__()
        self.gen = gen
        self.entropy_estimator = get_entropy_estimator(entropy_estimator)
        self.prior = self.set_prior(prior)
        self.transforms = transforms
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)
        self.sim = Simulator(self.transforms, self.diagnostics)
        self.discrepancy_function = get_loss_function(discrepancy_function)
        self.penalty_parameter = penalty_parameter

    def set_prior(self, prior):
        self.prior = prior
        if self.entropy_estimator is not None:
            self.entropy_estimator.prior = prior
        return self.prior

    def set_diagnostics(self, diagnostics: List[Type[nn.Module]]):
        self.diagnostics = diagnostics
        if self.diagnostics is None:
            self.diagnostics = [[]]
        self.sim = Simulator(self.transforms, self.diagnostics)
        return self.diagnostics

    def set_measurements(self, measurements: List[List[torch.Tensor]]):
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = [[]]
        return self.measurements

    def sample(self, n: int) -> torch.Tensor:
        """Sample n points."""
        return self.gen.sample(int(n))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability at x."""
        return self.gen.log_prob(x)

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and return the log probability at each point."""
        return self.gen.sample_and_log_prob(n)

    def sample_and_entropy(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and estimate the entropy from the samples."""
        x, log_prob = self.sample_and_log_prob(n)
        H = self.entropy_estimator(x, log_prob)
        return (x, H)

    def discrepancy_vector(self, predictions: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        discrepancy_vector = []
        for pred, meas in zip(unravel(predictions), unravel(self.measurements)):
            discrepancy_vector.append(self.discrepancy_function(pred, meas))
        return discrepancy_vector

    def loss(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Estimate the loss from a new batch.

        Parameters
        ----------
        batch_size : int
            Number of particles to samples.

        Returns
        -------
        L : tensor
            L = H + mu * |D|, where H is the entropy, D is the discrepancy vector, and |.| is
            the l1 norm.
        H : tensor
            Estimated entropy. If `self.prior` is not None, this is the negative relative entropy,
            or the KL divergence between the model and prior distribution. Otherwise we estimate
            the absolute entropy.
        D : tensor, shape (len(measurements) * len(diagnostics))
            The discrepancy vector.
        """
        x, H = self.sample_and_entropy(batch_size)
        predictions = self.sim.forward(x)
        D = self.discrepancy_vector(predictions)
        L = H + self.penalty_parameter * (sum(D) / len(D))
        return (L, H, D)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.gen.parameters()
    
    def save(self, path) -> None:
        state = {
            "gen": self.gen.state_dict(),
            "entropy_estimator": self.entropy_estimator,
            "prior": self.prior,
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "measurements": self.measurements,
        }
        torch.save(state, path)

    def load(self, path, device=None):
        state = torch.load(path, map_location=device)
        try:
            self.gen.load_state_dict(state["gen"])
        except RuntimeError:
            raise RuntimeError("Error loading generative model. Architecture mismatch?")

        self.entropy_estimator = state["entropy_estimator"]
        self.set_prior(state["prior"])
        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.measurements = state["measurements"]
        self.to(device)

    def to(self, device):
        if self.transforms is not None:
            for transform in self.transforms:
                transform = transform.to(device)
        if self.diagnostics is not None:
            for index in range(len(self.diagnostics)):
                for diagnostic in self.diagnostics[index]:
                    diagnostic = diagnostic.to(device)
        if self.measurements is not None:
            for index in range(len(self.measurements)):
                for measurement in self.measurements[index]:
                    measurement = measurement.to(device)
        if self.gen is not None:
            self.gen = self.gen.to(device)
        return self


