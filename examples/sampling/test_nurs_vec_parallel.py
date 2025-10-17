from mpi4py import MPI
import numpy as np
import nurs

# Import your vectorized NURS sampler here
# from nurs_module import nurs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global parameters
num_chains_total = 32  # total chains across all MPI processes
num_draws = 100
dim = 2
step_size = 0.1
max_doublings = 5
threshold = 1e-3

# Each process handles a subset of chains
chains_per_proc = num_chains_total // size
remainder = num_chains_total % size
if rank < remainder:
    local_chains = chains_per_proc + 1
    start_idx = rank * local_chains
else:
    local_chains = chains_per_proc
    start_idx = rank * local_chains + remainder

rng = np.random.default_rng(seed=rank)  # independent RNG per process
theta_init_local = rng.normal(size=(local_chains, dim))

# Define your log_prob_func (batch-compatible)
def log_prob_func(theta):
    # Example: standard normal
    return -0.5 * np.sum(theta**2, axis=1)


# Run the sampler locally on this process
draws_local, accepts_local, depths_local = nurs.nurs_vectorized(
    rng=rng,
    log_prob_func=log_prob_func,
    theta_init=theta_init_local,
    num_draws=num_draws,
    step_size=step_size,
    max_doublings=max_doublings,
    threshold=threshold,
)

# Gather results to root process
# 1D flatten for MPI communication, then reshape after gather
draws_local_flat = draws_local.reshape(num_draws, -1)
all_draws_flat = None
if rank == 0:
    all_draws_flat = np.empty((num_draws, num_chains_total * dim))
comm.Gather(draws_local_flat, all_draws_flat, root=0)

accepts_local_flat = accepts_local
all_accepts_flat = None
if rank == 0:
    all_accepts_flat = np.empty((num_draws, num_chains_total), dtype=int)
comm.Gather(accepts_local_flat, all_accepts_flat, root=0)

depths_local_flat = depths_local
all_depths_flat = None
if rank == 0:
    all_depths_flat = np.empty((num_draws, num_chains_total), dtype=int)
comm.Gather(depths_local_flat, all_depths_flat, root=0)

# Reshape results on root
if rank == 0:
    all_draws = all_draws_flat.reshape(num_draws, num_chains_total, dim)
    all_accepts = all_accepts_flat
    all_depths = all_depths_flat

    print("All draws shape:", all_draws.shape)
    print("Mean acceptance:", all_accepts.mean())
    print("Mean tree depth:", all_depths.mean())
