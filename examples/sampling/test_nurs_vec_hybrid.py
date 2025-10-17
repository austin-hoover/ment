from mpi4py import MPI
import numpy as np
import nurs


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----- Global sampler settings -----
num_chains_total = 100  # total chains across all MPI processes
num_draws = 10000 // size
dim = 2
step_size = 0.1
max_doublings = 5
threshold = 1e-3

# ----- Compute per-process chain allocation -----
chains_per_proc = num_chains_total // size
remainder = num_chains_total % size
if rank < remainder:
    local_chains = chains_per_proc + 1
    start_idx = rank * local_chains
else:
    local_chains = chains_per_proc
    start_idx = rank * local_chains + remainder

rng = np.random.default_rng(seed=rank)

# Initial theta for this process
theta_init_local = rng.normal(size=(local_chains, dim))

# ----- Log-probability function (batched) -----
def log_prob_func(theta):
    # Standard multivariate normal for demonstration
    return -0.5 * np.sum(theta**2, axis=1)


# ----- Run the sampler locally (vectorized for local chains) -----
draws_local, accepts_local, depths_local = nurs.nurs_vectorized(
    rng=rng,
    log_prob_func=log_prob_func,
    theta_init=theta_init_local,
    num_draws=num_draws,
    step_size=step_size,
    max_doublings=max_doublings,
    threshold=threshold,
)


# ----- Gather results using Gatherv (no flattening needed) -----
# Prepare counts and displacements for each process
send_counts_draws = np.array([num_draws * local_chains * dim], dtype=int)
recv_counts_draws = None
recv_displs_draws = None

if rank == 0:
    recv_counts_draws = np.zeros(size, dtype=int)
    for r in range(size):
        extra = 1 if r < remainder else 0
        recv_counts_draws[r] = num_draws * (chains_per_proc + extra) * dim
    recv_displs_draws = np.insert(np.cumsum(recv_counts_draws), 0, 0)[0:-1]

draws_flat_local = draws_local.ravel()
draws_flat_all = None
if rank == 0:
    draws_flat_all = np.empty(np.sum(recv_counts_draws), dtype=draws_local.dtype)

comm.Gatherv(
    draws_flat_local,
    [draws_flat_all, recv_counts_draws, recv_displs_draws, MPI.DOUBLE],
    root=0,
)

# Reshape gathered draws on root
if rank == 0:
    # Compute number of chains per rank for reconstruction
    chain_counts = [
        (chains_per_proc + (1 if r < remainder else 0)) for r in range(size)
    ]
    draws_list = []
    idx = 0
    for c in chain_counts:
        n = num_draws * c * dim
        draws_list.append(draws_flat_all[idx : idx + n].reshape(num_draws, c, dim))
        idx += n
    draws_all = np.concatenate(draws_list, axis=1)  # (num_draws, num_chains_total, dim)

    print("All draws shape:", draws_all.shape)
