import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


class Transform:
    def __init__(self) -> None:
        return
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        NotImplementedError

    def inverse(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearTransform(Transform):
    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__()
        self.matrix = matrix
        self.matrix_inv = np.linalg.inv(matrix)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.matrix.T)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.matrix_inv.T)


class ComposedTransform(Transform):
    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def forward(self, x: np.ndarray) -> np.ndarray:
        u = x
        for transform in transforms:
            u = transform(u)
        return u

    def inverse(self, u: np.ndarray) -> np.ndarray:
        x = u
        for transform in reversed(transforms):
            x = transform.inverse(x)
        return x


def forward(x, transforms, diagnostics):
    predictions = []
    for index, transform in enumerate(transforms):
        u = transform(x)
        predictions.append([diagnostic(u) for diagnostic in diagnostics[index]])
    return predictions


