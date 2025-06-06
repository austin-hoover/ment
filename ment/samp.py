import math
import time
import warnings
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
    merge: bool = True,
    allow_zero_start: bool = True,
    seed: int = None,
    verbose: int = 0,
    debug: bool = False,
) -> np.ndarray:
    """Vectorized Metropolis-Hastings.

    https://colindcarroll.com/2019/08/18/very-parallel-mcmc-sampling/

    Parameters
    ----------
    prob_func : Callable
        Function returning probability density p(x) at points x. The function must be
        vectorized so that x is a batch of points of shape (nchains, ndim).
    size : int
        The number of samples per chain (excluding burn-in).
    chains : int
        Number of sampling chains.
    burnin : int
        Number of burnin iterations (applies to each chain).
    start ; np.ndarray, shape
        An array of shape (chains, ndim) giving the starting point of each chain. All
        start points must be in regions of nonzero probability density.
    proposal_cov : ndarray
        We use a Gaussian proposal distribution centered on the current point in
        the random walk. This variable is the covariance matrix of the Gaussian
        distribution.
    merge : bool
        Whether to merge the sampling chains. If the chains are merged, the return
        array has shape (size * chains, ndim). Otherwise if has shape (size, chains, ndim).
    allow_zero_start : bool
        If False, raise an error if any of the chains starts at a point of zero density.
    seed : int
        Seed used in all random number generators.
    verbose : int
        Whether to show progress bar.
    debug : bool
        Whether to print iteration number and acceptance rate.

    Returns
    -------
    ndarray
        Sampled points with burn-in points discarded. Shape is (size, ndim) if merge=True
        or (size * chains, chains, ndim) if merge=False.
    """  
    debug = int(debug)
    rng = np.random.default_rng(seed)
    size = size + burnin

    # Initialize list of points. From now on we each "point" is really a batch of 
    # size (nchains, ndim). Burnin-points will be discarded later.
    points = np.empty((size, chains, ndim)) 

    # Sample proposal points from a Gaussian distribution. (The means will be updated
    # during the random walk.)
    proposal_mean = np.zeros(ndim)
    if proposal_cov is None:
        proposal_cov = np.eye(ndim)
        
    proposal_points = rng.multivariate_normal(
        proposal_mean, proposal_cov, size=(size - 1, chains)
    )

    # Set starting point for each chain. If none is provided, sample from the proposal
    # distribution centered at the origin.
    if start is None:
        start = rng.multivariate_normal(proposal_mean, proposal_cov, size=chains)
        start *= 0.50
        
    points[0] = start

    # Raise an error if any points give nan probability.
    prob = prob_func(points[0])
    if np.any(np.logical_or(np.equal(prob, 0.0), np.isnan(prob))):
        n = np.count_nonzero(np.isnan(prob))
        if n > 0:
            raise ValueError(f"Invalid starting point! p(x) = NaN at {n} points!")

    # Raise an error if any points start in a region of zero probability density.
    if not allow_zero_start:
        n = np.count_nonzero(prob == 0)
        if n > 0:
            raise ValueError(f"Invalid starting point! p(x) = 0 at {n} points!")

    # Execute random walks
    random_uniform = rng.uniform(size=(size - 1, chains))
    accept = np.zeros(chains)
    n_total_accepted = 0
    n_total = 0
    acceptance_rate = None

    for i in wrap_tqdm(range(1, size), verbose):
        proposal_point = points[i - 1] + proposal_points[i - 1]
        proposal_prob = prob_func(proposal_point)
        accept = proposal_prob > prob * random_uniform[i - 1]

        if i > burnin:
            n_total_accepted += np.count_nonzero(accept)
            n_total += chains
            acceptance_rate = n_total_accepted / n_total
            if debug > 1:
                print("debug {:05.0f} acceptance_rate={:0.4f}".format(i, acceptance_rate))

        points[i] = points[i - 1]
        points[i][accept] = proposal_point[accept]
        prob[accept] = proposal_prob[accept]

    points = points[burnin:]

    if debug:
        print("debug acceptance rate =", acceptance_rate)
        for axis in range(ndim):
            x_chain_stds = [np.std(chain[:, axis]) for chain in points]
            x_chain_avgs = [np.mean(chain[:, axis]) for chain in points]
            print(f"debug axis={axis} between-chain avg(x_chain_std) =", np.mean(x_chain_stds))
            print(f"debug axis={axis} between-chain std(x_chain_std) =", np.std(x_chain_stds))
            print(f"debug axis={axis} between-chain avg(x_chain_avg) =", np.mean(x_chain_avgs))
            print(f"debug axis={axis} between-chain std(x_chain_avg) =", np.std(x_chain_avgs))
    
            x_avg = np.mean(np.hstack([chain[:, axis] for chain in points]))
            x_std = np.std( np.hstack([chain[:, axis] for chain in points]))
            print(f"debug axis={axis} x_std =", x_std)
            print(f"debug axis={axis} x_avg =", x_avg)


    # Option to return unmerged chains:
    if merge:
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
    
    return points


class Sampler:
    def __init__(self, ndim: int, seed: int = None, verbose: int = 0, debug: bool = False) -> None:
        self.ndim = ndim
        self.seed = seed
        self.verbose = verbose
        self.debug = debug
        
        self.rng = np.random.default_rng(seed)

    def sample(self, prob_func: Callable, size: int) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, prob_func: Callable, size: int) -> np.ndarray:
        return self.sample(prob_func, size)


class MetropolisHastingsSampler(Sampler):
    def __init__(
        self,
        chains: int = 1,
        burnin: int = 1000,
        start: np.ndarray = None,
        proposal_cov: np.ndarray = None,
        allow_zero_start: bool = True,
        shuffle: bool = False,
        noise_scale: float = None,
        noise_type: str = "uniform",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chains = chains
        self.burnin = burnin
        self.start = start
        self.proposal_cov = proposal_cov
        self.shuffle = shuffle
        self.allow_zero_start = allow_zero_start

        self.noise_scale = noise_scale
        self.noise_type = noise_type

    def add_noise(self, x: np.ndarray) -> np.ndarray:
        x_add = np.zeros(x.shape)
        if self.noise_type == "uniform":
            x_add = self.rng.uniform(size=x.shape) - 0.5
            x_add *= self.noise_scale
        elif self.noise_type == "gaussian":
            x_add = self.rng.normal(scale=self.noise_scale, size=x.shape)
        return x + x_add            
                
    def sample(self, prob_func: Callable, size: int) -> np.ndarray:
        x = sample_metropolis_hastings(
            prob_func,
            ndim=self.ndim,
            size=int(math.ceil(size / float(self.chains))),
            chains=self.chains,
            burnin=self.burnin,
            start=self.start,
            proposal_cov=self.proposal_cov,
            merge=True,
            allow_zero_start=self.allow_zero_start,
            verbose=self.verbose,
            debug=self.debug,
            seed=self.seed,
        )        
        if self.shuffle:
            self.rng.shuffle(x)
        if self.noise_scale:
            x = self.add_noise(x)
        return x[:size]


class GridSampler(Sampler):
    def __init__(
        self,
        grid_limits: list[tuple[float, float]],
        grid_shape: tuple[int, ...],
        noise: float = 0.0,
        store_grid_points: bool = True,
        **kwargs
    ) -> None:
        super().__init__(ndim=len(grid_limits), **kwargs)
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits
        self.noise = noise
        self.store_grid_points = store_grid_points

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

    def sample(self, prob_func: Callable, size: int) -> np.ndarray:
        prob = prob_func(self.get_grid_points())
        prob = np.reshape(prob, self.grid_shape)
        x = sample_hist(prob, self.grid_edges, size, noise=self.noise, rng=self.rng)
        return x


class SliceGridSampler(Sampler):
    def __init__(
        self,
        grid_limits: list[tuple[float, float]],
        grid_shape: tuple[int, ...],
        proj_dim: int = 2,
        int_size: int = 10000,
        int_method: str = "grid",
        int_batches: int = 1,
        noise: float = 0.0,
        **kwargs
    ):
        super().__init__(ndim=len(grid_limits), **kwargs)
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits
        self.ndim = len(grid_limits)
        
        self.proj_dim = proj_dim
        self.samp_dim = self.ndim - self.proj_dim
        self.noise = noise

        self.grid_edges = [
            np.linspace(
                self.grid_limits[axis][0],
                self.grid_limits[axis][1],
                self.grid_shape[axis] + 1,
            )
            for axis in range(self.ndim)
        ]
        self.grid_coords = [0.5 * (e[:-1] + e[1:]) for e in self.grid_edges]

        # Projection grid
        self.proj_axis = tuple(range(self.proj_dim))
        self.proj_grid_shape = self.grid_shape[: self.proj_dim]
        self.proj_grid_edges = self.grid_edges[: self.proj_dim]
        self.proj_grid_coords = self.grid_coords[: self.proj_dim]
        self.proj_grid_points = get_grid_points(self.proj_grid_coords)

        # Sampling grid
        self.samp_axis = tuple(range(self.proj_dim, self.ndim))
        self.samp_grid_shape = self.grid_shape[self.proj_dim :]
        self.samp_grid_edges = self.grid_edges[self.proj_dim :]
        self.samp_grid_coords = self.grid_coords[self.proj_dim :]
        self.samp_grid_points = get_grid_points(self.samp_grid_coords)

        # Integration limits (integration axis = sampling axis)
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
            int_coords = [np.linspace(xmin, xmax, int_res) for (xmin, xmax) in self.int_limits]
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
        return int_points[: self.int_size]

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

    def sample(self, prob_func: Callable, size: int) -> np.ndarray:
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
                    x_loc[:, axis] += (
                        self.noise * 0.5 * self.rng.uniform(-delta, delta, size=size_loc)
                    )

            # Set evaluation points on projected subpsace.
            for axis, index in enumerate(indices):
                self.eval_points[:, axis] = self.proj_grid_coords[axis][index]

            # Evaluate sliced density rho(x_samp | x_proj).
            prob = prob_func(self.eval_points)
            prob = np.reshape(prob, self.samp_grid_shape)  # could cut this step
            x_loc[:, self.samp_axis] = sample_hist(
                prob,
                self.samp_grid_edges,
                size_loc,
                noise=self.noise,
                rng=self.rng,
            )

            x.append(x_loc)

        # Stack points from all bins, then shuffle.
        x = np.vstack(x)
        self.rng.shuffle(x)
        return x
