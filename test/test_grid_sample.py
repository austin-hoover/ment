import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import psdist as ps
import psdist.visualization as psv
import scipy.interpolate

sys.path.append("..")
import ment.samp


def func(x: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(x, axis=1)
    prob = np.exp(-0.5 * r**2)
    return prob


ndim = 4
grid_shape = [75] * ndim
grid_limits = [(-5.0, 5.0)] * ndim
n_samples = 100_000
noise = 0.5


sampler = ment.samp.GridSampler(
    grid_limits=grid_limits,
    grid_shape=grid_shape,
    noise=noise,
)

print("GS")
start_time = time.time()

x = sampler(func, n_samples)

print("time =", time.time() - start_time)


proj_dim = 1
sampler = ment.samp.SliceGridSampler(
    grid_limits=grid_limits,
    grid_shape=grid_shape,
    proj_dim=proj_dim,
    int_size=(15 ** proj_dim),
    int_method="grid",
    int_batches=1,
    noise=noise,
    verbose=True,
)

print("SGS")
start_time = time.time()

x = sampler(func, n_samples)

print("time =", time.time() - start_time)


