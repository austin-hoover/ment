import itertools
import torch


class RegularGridInterpolator:
    """Interpolate points on a regular grid.

    https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py
    """

    def __init__(
        self, coords: list[torch.Tensor], values: torch.Tensor, fill_value: float = 0.0
    ) -> None:
        self.coords = coords
        self.values = values
        self.fill_value = torch.tensor(fill_value)

        if type(self.coords) is torch.Tensor:
            self.coords = [
                self.coords,
            ]

        assert isinstance(self.coords, tuple) or isinstance(self.coords, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.ndim = len(self.coords)

        assert len(self.ms) == self.ndim

        for i, p in enumerate(self.coords):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, new_points: torch.Tensor) -> torch.Tensor:
        assert self.coords is not None
        assert self.values is not None

        if new_points.ndim == 1:
            new_points = new_points[:, None]

        new_points_t = new_points.T

        assert new_points_t.shape[0] == self.ndim

        K = new_points_t.shape[1]
        for x in new_points_t:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.coords, new_points_t):
            idx_right = torch.bucketize(x.contiguous(), p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.0
            dist_right[dist_right < 0] = 0.0
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.0

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.0
        for indexer in itertools.product([0, 1], repeat=self.ndim):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[tuple(as_s)] * torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        result = numerator / denominator

        # Handle bounds
        out_of_bounds = torch.zeros(
            new_points_t.shape[1], dtype=torch.bool, device=self.values.device
        )
        for x, c in zip(new_points_t, self.coords):
            out_of_bounds = out_of_bounds | (x < c[0]) | (x > c[-1])
        result[out_of_bounds] = self.fill_value

        return result


class RegularGridInterpolationStencil:
    """Caches interpolation indices and weights for fixed points."""

    def __init__(self, coords: list[torch.Tensor], points: torch.Tensor) -> None:
        if type(coords) is torch.Tensor:
            coords = [coords]
        if points.ndim == 1:
            points = points[:, None]

        self.coords = coords
        self.ndim = len(coords)
        self.idxs = []
        self.weights = []
        self.valid = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

        points_t = points.T
        for coord, x in zip(coords, points_t):
            coord = coord.to(device=x.device, dtype=x.dtype)
            idx_right = torch.bucketize(x.contiguous(), coord)
            idx_right[idx_right >= coord.shape[0]] = coord.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, coord.shape[0] - 1)

            dist_left = x - coord[idx_left]
            dist_right = coord[idx_right] - x
            dist_left[dist_left < 0] = 0.0
            dist_right[dist_right < 0] = 0.0

            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.0

            denom = dist_left + dist_right
            self.idxs.append((idx_left, idx_right))
            self.weights.append((dist_right / denom, dist_left / denom))
            self.valid = self.valid & (x >= coord[0]) & (x <= coord[-1])

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        result = 0.0
        for indexer in itertools.product([0, 1], repeat=self.ndim):
            idx = [idx_pair[bit] for bit, idx_pair in zip(indexer, self.idxs)]
            weights = [
                weight_pair[bit] for bit, weight_pair in zip(indexer, self.weights)
            ]
            result += values[tuple(idx)] * torch.prod(torch.stack(weights), dim=0)
        return result * self.valid.to(dtype=values.dtype)
