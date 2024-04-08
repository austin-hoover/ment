"""Experimental sampling module using Jax.

`get_grid_points_jax` is much faster for large grids. But `sample_bins` is slower and throws the following warnings. I think these are specific to jax-metal.

warning: loc("jit(sample_bins_jax)/jit(main)/select_n"(callsite("sample_bins_jax"("/Users/46h/repo/MENT/ment/samp_jax.py":29:0) at "<module>"("/Users/46h/repo/MENT/ment/samp_jax.py":112:0)))): 'anec.not_equal_zero' op Invalid configuration for the following reasons: Tensor dimensions N1D1C1H1W64000000 are not within supported range, N[1-65536]D[1-16384]C[1-65536]H[1-16384]W[1-16384].

warning: loc(callsite(callsite("jit(sample_bins_jax)/jit(main)/jit(_take)/jit(_where)/select_n"(callsite("sample_bins_jax"("/Users/46h/repo/MENT/ment/samp_jax.py":32:0) at "<module>"("/Users/46h/repo/MENT/ment/samp_jax.py":112:0))) at "jit(sample_bins_jax)/jit(main)/jit(_take)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False, False) name=_where keep_unused=False inline=False]"(callsite("sample_bins_jax"("/Users/46h/repo/MENT/ment/samp_jax.py":32:0) at "<module>"("/Users/46h/repo/MENT/ment/samp_jax.py":112:0)))) at "jit(sample_bins_jax)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=_take keep_unused=False inline=False]"(callsite("sample_bins_jax"("/Users/46h/repo/MENT/ment/samp_jax.py":32:0) at "<module>"("/Users/46h/repo/MENT/ment/samp_jax.py":112:0))))): 'anec.not_equal_zero' op Invalid configuration for the following reasons: Tensor dimensions N1D1C1H1W1000000 are not within supported range, N[1-65536]D[1-16384]C[1-65536]H[1-16384]W[1-16384].

warning: loc(callsite(callsite("jit(sample_bins_jax)/jit(main)/jit(floor_divide)/jit(_where)/select_n"(callsite("sample_bins_jax"("/Users/46h/repo/MENT/ment/samp_jax.py":29:0) at "<module>"("/Users/46h/repo/MENT/ment/samp_jax.py":112:0))) at "jit(sample_bins_jax)/jit(main)/jit(floor_divide)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False, False) name=_where keep_unused=False inline=False]"(callsite("sample_bins_jax"("/Users/46h/repo/MENT/ment/samp_jax.py":29:0) at "<module>"("/Users/46h/repo/MENT/ment/samp_jax.py":112:0)))) at "jit(sample_bins_jax)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=floor_divide keep_unused=False inline=False]"(callsite("sample_bins_jax"("/Users/46h/repo/MENT/ment/samp_jax.py":29:0) at "<module>"("/Users/46h/repo/MENT/ment/samp_jax.py":112:0))))): 'anec.not_equal_zero' op Invalid configuration for the following reasons: Tensor dimensions N1D1C1H1W64000000 are not within supported range, N[1-65536]D[1-16384]C[1-65536]H[1-16384]W[1-16384].
"""
import functools
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def get_grid_points(*grid_coords: np.ndarray) -> np.ndarray:
    return np.stack([C.ravel() for C in np.meshgrid(*grid_coords, indexing="ij")], axis=-1)


@jax.jit
def get_grid_points_jax(*grid_coords: ArrayLike) -> Array:
    return jnp.stack([jnp.ravel(C) for C in jnp.meshgrid(*grid_coords, indexing="ij")], axis=-1)


def sample_bins(values: np.ndarray, size: int) -> np.ndarray:
    pdf = np.ravel(values)
    idx = np.flatnonzero(pdf)
    pdf = pdf[idx]
    pdf = pdf / np.sum(pdf)
    return np.random.choice(idx, size, replace=True, p=pdf)


@functools.partial(jax.jit, static_argnames=["size", "seed"])
def sample_bins_jax(values: ArrayLike, size: int, seed: int = 758493) -> Array:
    pdf = jnp.ravel(values)
    idx = jnp.flatnonzero(pdf, size=values.size)
    p = pdf[idx] / jnp.sum(pdf[idx])
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, idx, shape=(size,), replace=True, p=p)


def sample_hist(
    values: np.ndarray, 
    edges: list[np.ndarray], 
    size: int = 100, 
    noise: float = 0.0, 
) -> np.ndarray:
    idx = sample_bins(values, size)
    idx = np.unravel_index(idx, shape=values.shape)
    x = np.zeros((size, values.ndim))
    for axis in range(x.shape[1]):
        lb = edges[axis][idx[axis]]
        ub = edges[axis][idx[axis] + 1]
        x[:, axis] = np.random.uniform(lb, ub)
        if noise:
            delta = ub - lb
            x[:, axis] += 0.5 * self.noise * np.random.uniform(-delta, delta)
    x = np.squeeze(x)
    return x

    

if __name__ == "__main__":
    
    import time
    
    print("get_grid_points")
    dim = 4
    res = 50
    
    print("numpy")
    start_time = time.time()
    grid_coords = dim * [np.linspace(0.0, 1.0, res)]
    grid_points = get_grid_points(*grid_coords)
    print(f"time = {time.time() - start_time}")
    print(f"grid_points.nbytes = {grid_points.nbytes:0.3e}")
    
    print("jax")
    start_time = time.time()
    grid_coords = dim * [jnp.linspace(0.0, 1.0, res)]
    grid_points = get_grid_points_jax(*grid_coords)
    print(f"time = {time.time() - start_time}")

    print()
    print("sample_bins")
    n_samples = 1_000_000
    grid_shape = 6 * [20]
    
    print("numpy")
    values = np.random.uniform(size=grid_shape)
    start_time = time.time()
    idx = sample_bins(values, n_samples)
    print(f"time = {time.time() - start_time}")
    
    print("jax")
    key = jax.random.PRNGKey(0)
    values = jax.random.uniform(key, shape=grid_shape)
    start_time = time.time()
    idx = sample_bins_jax(values, n_samples)
    print(f"time = {time.time() - start_time}")

