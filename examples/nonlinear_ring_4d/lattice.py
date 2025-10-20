import math
import torch

from ment.sim import Transform


class AxiallySymmetricNonlinearKick(Transform):
    def __init__(
        self, alpha: float, beta: float, phi: float, A: float, E: float, T: float
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.A = A
        self.E = E
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 2] ** 2)
        t = torch.arctan2(x[:, 2], x[:, 0])

        alpha = self.alpha
        beta = self.beta
        phi = self.phi
        E = self.E
        A = self.A
        T = self.T

        dr = -(1.0 / (beta * math.sin(phi))) * ((E * r) / (A * r**2 + T)) - (
            (2.0 * r) / (beta * math.tan(phi))
        )

        x_out = torch.clone(x)
        x_out[:, 1] += dr * torch.cos(t)
        x_out[:, 3] += dr * torch.sin(t)
        return x_out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x[:, 1] *= -1.0
        X = self.forward(X)
        X[:, 1] *= -1.0
        return X
