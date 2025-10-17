import numpy as np
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        r = np.sqrt(x[:, 0] ** 2 + x[:, 2] ** 2)
        t = np.arctan2(x[:, 2], x[:, 0])

        alpha = self.alpha
        beta = self.beta
        phi = self.phi
        E = self.E
        A = self.A
        T = self.T

        dr = -(1.0 / (beta * np.sin(phi))) * ((E * r) / (A * r**2 + T)) - (
            (2.0 * r) / (beta * np.tan(phi))
        )

        x_out = np.copy(x)
        x_out[:, 1] += dr * np.cos(t)
        x_out[:, 3] += dr * np.sin(t)
        return x_out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x[:, 1] *= -1.0
        X = self.forward(X)
        X[:, 1] *= -1.0
        return X
