import math
import time
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
from tqdm import tqdm

from .grid import coords_to_edges
from .grid import edges_to_coords
from .grid import get_grid_points
from .utils import wrap_tqdm


def tqdm_wrapper(iterable, verbose=False):
    return tqdm(iterable) if verbose else iterable
    

def sample_bins(values: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(values.size, size, replace=True, p=(np.ravel(values) / np.sum(values)))
    

def sample_hist(
    values: np.ndarray, 
    edges: list[np.ndarray], 
    size: int,
    rng: np.random.Generator,
    noise: float = 0.0, 
) -> np.ndarray:

    idx = sample_bins(values, size, rng=rng)
    idx = np.unravel_index(idx, shape=values.shape)
    
    x = np.zeros((size, values.ndim))
    for axis in range(x.shape[1]):
        lb = edges[axis][idx[axis]]
        ub = edges[axis][idx[axis] + 1]
        x[:, axis] = rng.uniform(lb, ub)
        if noise:
            delta = ub - lb
            x[:, axis] += 0.5 * noise * rng.uniform(-delta, delta)
    x = np.squeeze(x)
    return x
    

def sample_hist_and_rebin(values: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    idx = sample_bins(values, size, rng=rng)
    edges = np.arange(values.size + 1) - 0.5
    values_out, _ = np.histogram(idx, bins=edges, density=False)
    values_out = np.reshape(values_out, values.shape)
    return values_out
    

def sample_metropolis_hastings(
    prob_func: Callable, 
    ndim: int, 
    size: int,
    chains: int = 1,
    burnin: int = 10_000,
    start: np.ndarray = None,
    proposal_cov: np.ndarray = None,
    pad: float = 1.00e-14,
    merge: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Vectorized Metropolis-Hastings.

    https://colindcarroll.com/2019/08/18/very-parallel-mcmc-sampling/
    """
    def log_prob_func(x):
        return np.log(prob_func(x) + pad)

    if proposal_cov is None:
        proposal_cov = np.eye(ndim)
    proposal_mean = np.zeros(ndim)

    if start is None:
        ub = 2.0 * np.sqrt(np.diagonal(proposal_cov))
        lb = -ub
        start = np.random.uniform(lb, ub, size=(chains, ndim))

    size += burnin

    points = np.empty((size, chains, ndim))
    points[0] = start
    current_log_prob = log_prob_func(start)

    proposals = np.random.multivariate_normal(proposal_mean, proposal_cov, size=(size - 1, chains))
    log_unifs = np.log(np.random.rand(size - 1, chains))

    for i in wrap_tqdm(range(1, size), verbose):
        proposal = proposals[i - 1] + points[i - 1]
        proposal_log_prob = log_prob_func(proposal)
        accept = log_unifs[i - 1] < (proposal_log_prob - current_log_prob)
    
        points[i] = points[i - 1]
        points[i][accept] = proposal[accept]

        current_log_prob[accept] = proposal_log_prob[accept]

    points = points[burnin:]
    if merge:
        shape = points.shape
        shape = (shape[0] * shape[1], shape[2])
        points = points.reshape(shape)
    return points


class MetropolisHastingsSampler:
    def __init__(
        self,
        ndim: int, 
        chains: int = 10,
        burnin: int = 1_000,
        start: np.ndarray = None,
        proposal_cov: np.ndarray = None,
        shuffle: bool = False,
        verbose: bool = False,
        pad: float = 1.00e-14,
    ) -> None:
        self.ndim = ndim
        self.chains = chains
        self.burnin = burnin
        self.start = start
        self.proposal_cov = proposal_cov
        self.shuffle = shuffle
        self.verbose = verbose
        self.pad = pad

    def __call__(self, prob_func: Callable, size: int) -> np.ndarray:
        x = sample_metropolis_hastings(
            prob_func, 
            ndim=self.ndim, 
            size=(size // self.chains), 
            chains=self.chains,
            burnin=self.burnin,
            start=self.start,
            proposal_cov=self.proposal_cov,
            merge=True,
            verbose=self.verbose,
            pad=self.pad
        )  
        if self.shuffle:
            np.random.shuffle(x)
        return x


class GridSampler:
    def __init__(
        self,
        grid_limits: list[tuple[float, float]],
        grid_shape: tuple[int, ...],
        noise: float = 0.0,
        store_grid_points: bool = True,
        seed: int = None
    ) -> None:
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits
        self.ndim = len(grid_limits)
        self.noise = noise
        self.store_grid_points = store_grid_points
        self.rng = np.random.default_rng(seed)
        
        self.grid_edges = [
            np.linspace(
                self.grid_limits[axis][0],
                self.grid_limits[axis][1],
                self.grid_shape[axis] + 1,
            )
            for axis in range(self.ndim)
        ]
        self.grid_coords = [edges_to_coords(e) for e in self.grid_edges]
        self.grid_points = None

    def get_grid_points(self) -> np.ndarray:
        if self.grid_points is not None:
            return self.grid_points
        grid_points = get_grid_points(self.grid_coords)        
        if self.store_grid_points:
            self.grid_points = grid_points
        return grid_points

    def __call__(self, prob_func: Callable, size: int) -> np.ndarray:
        prob = prob_func(self.get_grid_points()) 
        prob = np.reshape(prob, self.grid_shape)
        x = sample_hist(prob, self.grid_edges, size, noise=self.noise, rng=self.rng)
        return x


class SliceGridSampler:
    def __init__(
        self,
        grid_limits: list[tuple[float, float]],
        grid_shape: tuple[int, ...],
        proj_dim: int = 2,
        int_size: int = 10000,
        int_method: str = "grid",
        int_batches: int = 1,
        noise: float = 0.0,
        verbose: bool = False,
        seed: int = None,
    ):
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits
        self.ndim = len(grid_limits)
        self.proj_dim = proj_dim
        self.samp_dim = self.ndim - self.proj_dim
        self.noise = noise
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.grid_edges = [
            np.linspace(
                self.grid_limits[axis][0],
                self.grid_limits[axis][1],
                self.grid_shape[axis] + 1,
            )
            for axis in range(self.ndim)
        ]
        self.grid_coords = [0.5 * (e[:-1] + e[1:]) for e in self.grid_edges]

        # Projection grid.
        self.proj_axis = tuple(range(self.proj_dim))
        self.proj_grid_shape = self.grid_shape[: self.proj_dim]
        self.proj_grid_edges = self.grid_edges[: self.proj_dim]
        self.proj_grid_coords = self.grid_coords[: self.proj_dim]
        self.proj_grid_points = get_grid_points(self.proj_grid_coords)

        # Sampling grid.
        self.samp_axis = tuple(range(self.proj_dim, self.ndim))
        self.samp_grid_shape = self.grid_shape[self.proj_dim :]
        self.samp_grid_edges = self.grid_edges[self.proj_dim :]
        self.samp_grid_coords = self.grid_coords[self.proj_dim :]
        self.samp_grid_points = get_grid_points(self.samp_grid_coords)

        # Integration limits (integration axis = sampling axis).
        self.int_size = int_size
        self.int_method = int_method
        self.int_batches = int_batches
        self.int_limits = self.grid_limits[self.proj_dim :]
        self.int_axis = self.samp_axis
        self.int_dim = self.samp_dim
        self.int_points = None

        # We will evaluate the function on the sampling grid. The first `proj_dim`
        # dimensions are the projected coordinates.
        self.eval_points = np.zeros((math.prod(self.samp_grid_shape), self.ndim))
        self.eval_points[:, self.samp_axis] = get_grid_points(self.samp_grid_coords)

    def get_int_points(self) -> np.ndarray:
        int_points = None
        if self.int_method == "grid":
            int_res = self.int_size ** (1.0 / self.int_dim)
            int_res = math.ceil(int_res)
            int_res = int(int_res)
            int_coords = [
                np.linspace(xmin, xmax, int_res)
                for (xmin, xmax) in self.int_limits
            ]
            int_points = get_grid_points(int_coords)
        elif self.int_method == "uniform":
            int_points = np.zeros((self.int_size, self.int_dim))
            for axis, (xmin, xmax) in zip(self.int_axis, self.int_limits):
                int_points[:, axis] = self.rng.uniform(xmin, xmax, size=self.int_size)
                
        elif self.int_method == "gaussian":
            int_points = np.zeros((self.int_size, self.int_dim))
            scale = [xmax for (xmin, xmax) in self.int_limits]
            for axis, (xmin, xmax) in zip(self.int_axis, self.int_limits):
                int_points[:, axis] = 0.75 * self.rng.uniform(xmin, xmax, size=self.int_size)
        else:
            raise ValueError("Invalid int_method")  
        return int_points[:self.int_size]

    def project(self, func: Callable) -> np.ndarray:
        """Project function onto onto lower dimensional plane."""
        rho = np.zeros(self.proj_grid_points.shape[0])
        x = np.zeros((self.int_size, self.ndim))
        for _ in range(self.int_batches):
            x[:, self.int_axis] = self.get_int_points()
            for i in tqdm_wrapper(range(rho.shape[0]), self.verbose):
                x[:, self.proj_axis] = self.proj_grid_points[i, :]
                rho[i] += np.sum(func(x))
        rho = np.reshape(rho, self.proj_grid_shape)
        return rho

    def __call__(self, prob_func: Callable, size: int) -> np.ndarray:
        # Compute projection and resample to find number of particles in each projected bin.
        if self.verbose:
            print("Projecting")
        proj = self.project(prob_func)
        proj = proj / np.sum(proj)
        sizes_loc = sample_hist_and_rebin(proj, size, rng=self.rng)
          
        # Sample the remaining coordinates from within each bin the projected subspace.
        if self.verbose:
            print("Sampling")

        x = []
        for indices in tqdm_wrapper(np.ndindex(*proj.shape), self.verbose):
            size_loc = sizes_loc[indices]
            if size_loc == 0:
                continue

            # Sample from uniform distribution over this bin in the projected space.
            x_loc = np.zeros((size_loc, self.ndim))
            for axis, index in enumerate(indices):
                lb = self.proj_grid_edges[axis][index]
                ub = self.proj_grid_edges[axis][index + 1]
                x_loc[:, axis] = self.rng.uniform(lb, ub, size=size_loc)
                if self.noise:
                    delta = ub - lb
                    x_loc[:, axis] += self.noise * 0.5 * self.rng.uniform(-delta, delta, size=size_loc)

            # Set evaluation points on projected subpsace.
            for axis, index in enumerate(indices):
                self.eval_points[:, axis] = self.proj_grid_coords[axis][index]

            # Evaluate sliced density rho(x_samp | x_proj).
            prob = prob_func(self.eval_points) 
            prob = np.reshape(prob, self.samp_grid_shape)  # could cut this step
            x_loc[:, self.samp_axis] = sample_hist(
                prob, self.samp_grid_edges, size_loc, noise=self.noise, rng=self.rng,
            )
            
            x.append(x_loc)

        # Stack points from all bins, then shuffle.
        x = np.vstack(x)
        self.rng.shuffle(x)
        return x
