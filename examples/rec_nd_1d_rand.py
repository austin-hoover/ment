#!/usr/bin/env python
# coding: utf-8

# # N:1 MENT — random projections

# In[ ]:


import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import ment
from ment.train.plot import Plotter
from ment.train.plot import PlotDistCorner
from ment.train.plot import PlotProj1D
from ment.utils import unravel

plt.style.use("../style.mplstyle")


# ## Settings

# In[ ]:


dist_name = "gaussian-mixture"
ndim = 6
xmax = 3.5
seed = 12345


# ## Source distribution

# In[ ]:


dist = ment.dist.get_dist(dist_name, ndim=ndim, seed=seed)
x_true = dist.sample(1_000_000)


# In[ ]:


limits = ndim * [(-xmax, xmax)]

grid = ment.train.plot.CornerGrid(ndim, figsize=(ndim * 1.4, ndim * 1.4))
grid.plot(x_true, limits=limits, bins=64)
plt.show()


# ## Data generation

# In[ ]:


nmeas = 10
nbins = 64
blur = 0.0


# Create phase space transformations.

# In[ ]:


class ProjectionTransform:
    def __init__(self, direction: torch.Tensor) -> None:
        self.direction = direction

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        return torch.sum(x * self.direction, axis=1)[:, None]


# In[ ]:


transforms = []
for _ in range(nmeas):
    direction = torch.randn(ndim)
    direction = direction / torch.linalg.norm(direction)
    transform = ProjectionTransform(direction)
    transforms.append(transform)


# Create histogram diagnostics.

# In[ ]:


axis_proj = axis_meas = 0
bin_edges = torch.linspace(-xmax, xmax, nbins + 1)

diagnostics = []
for transform in transforms:
    diagnostic = ment.Histogram1D(
        axis=axis_meas,
        edges=bin_edges,
    )
    diagnostics.append([diagnostic])


# Generate data from the source distribution.

# In[ ]:


projections = ment.simulate_with_diag_update(
    x_true,
    transforms,
    diagnostics,
    thresh=5.00e-03,
)


# ## Reconstruction model

# In[ ]:


prior = ment.GaussianPrior(ndim=ndim, scale=1.0)


# In[ ]:


samp_method = "mcmc"

if samp_method == "grid":
    samp_grid_res = 32
    samp_noise = 0.5
    samp_grid_shape = ndim * [samp_grid_res]
    samp_grid_limits = limits

    sampler = ment.samp.GridSampler(
        grid_limits=samp_grid_limits,
        grid_shape=samp_grid_shape,
        noise=samp_noise,
    )

elif samp_method == "mcmc":
    samp_burnin = 500
    samp_chains = 1000
    samp_prop_cov = torch.eye(ndim) * (0.5**2)
    samp_start = torch.randn(samp_chains, ndim) * 0.5

    sampler = ment.MetropolisHastingsSampler(
        ndim=ndim,
        start=samp_start,
        proposal_cov=samp_prop_cov,
        burnin=samp_burnin,
        shuffle=True,
        verbose=1,
        noise=0.10,  # slight smoothing
        noise_type="gaussian",
    )

else:
    raise ValueError


# In[ ]:


model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=sampler,
    nsamp=100_000,
    mode="forward",
    verbose=True,
)


# ## Training

# In[ ]:


plot_nsamp = x_true.shape[0]


# In[ ]:


plot_model = Plotter(
    n_samples=plot_nsamp,
    plot_proj=[
        PlotProj1D(log=False),
    ],
    plot_dist=[
        PlotDistCorner(
            fig_kws=dict(figsize=(ndim * 1.4, ndim * 1.4)),
            limits=(ndim * [(-xmax, xmax)]),
            bins=64,
        ),
    ],
)

eval_model = ment.train.Evaluator(nsamp=plot_nsamp)


# In[ ]:


trainer = ment.train.Trainer(
    model,
    plot_func=plot_model,
    eval_func=eval_model,
    notebook=True,
)

trainer.train(iters=3, lr=0.95)


# ## Evaluate

# In[ ]:


x_pred = model.unnormalize(model.sample(1_000_000))


# In[ ]:


grid = ment.train.plot.CornerGrid(ndim, figsize=(ndim * 1.4, ndim * 1.4))
for i, x in enumerate([x_true, x_pred]):
    color = ["black", "red"][i]
    grid.plot(
        x,
        limits=limits,
        bins=64,
        proc_kws=dict(scale="max", blur=1.0),
        kind="contour",
        colors=color,
        diag_kws=dict(color=color, kind="line"),
        levels=np.linspace(0.01, 1.0, 7),
    )
plt.show()


# In[ ]:
