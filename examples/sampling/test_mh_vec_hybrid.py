import numpy as np
from mpi4py import MPI
import tqdm


def metropolis_hastings_batch(rng, log_prob_func, theta_init, num_draws, step_size):
    """
    Vectorized Metropolis-Hastings for multiple chains.
    theta_init: (num_chains, dim)
    """
    num_chains, dim = theta_init.shape

    draws = np.zeros((num_draws, num_chains, dim))
    accepts = np.zeros((num_draws, num_chains), dtype=int)

    draws[0] = theta_init
    lp_current = log_prob_func(theta_init)

    for t in tqdm.tqdm(range(1, num_draws), initial=1, total=num_draws):
        # Propose
        proposal = draws[t - 1] + rng.normal(scale=step_size, size=(num_chains, dim))
        lp_proposal = log_prob_func(proposal)

        # Acceptance
        accept_prob = np.minimum(1.0, np.exp(lp_proposal - lp_current))
        accept = rng.random(num_chains) < accept_prob
        draws[t] = np.where(accept[:, None], proposal, draws[t - 1])
        lp_current = np.where(accept, lp_proposal, lp_current)
        accepts[t] = accept

    return draws, accepts


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global settings
num_chains_total = 100  # total chains
num_draws = 100_000 // size
dim = 2
step_size = 0.1

# Split chains among processes
chains_per_proc = num_chains_total // size
remainder = num_chains_total % size
if rank < remainder:
    local_chains = chains_per_proc + 1
else:
    local_chains = chains_per_proc

rng = np.random.default_rng(seed=rank)
theta_init_local = rng.normal(size=(local_chains, dim))

# Example log-probability function
def log_prob_func(theta):
    # Standard multivariate normal
    return -0.5 * np.sum(theta**2, axis=1)


# Run MH sampler locally
draws_local, accepts_local = metropolis_hastings_batch(
    rng, log_prob_func, theta_init_local, num_draws, step_size
)

# Gather results on root
# Flatten arrays for MPI communication
draws_local_flat = draws_local.ravel()
draws_all_flat = None
recv_counts = None
recv_displs = None

if rank == 0:
    recv_counts = []
    for r in range(size):
        c = chains_per_proc + (1 if r < remainder else 0)
        recv_counts.append(num_draws * c * dim)
    recv_counts = np.array(recv_counts, dtype=int)
    recv_displs = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]
    draws_all_flat = np.empty(np.sum(recv_counts), dtype=draws_local.dtype)

comm.Gatherv(
    draws_local_flat, [draws_all_flat, recv_counts, recv_displs, MPI.DOUBLE], root=0
)

# Reshape on root
if rank == 0:
    chain_counts = [chains_per_proc + (1 if r < remainder else 0) for r in range(size)]
    draws_list = []
    idx = 0
    for c in chain_counts:
        n = num_draws * c * dim
        draws_list.append(draws_all_flat[idx : idx + n].reshape(num_draws, c, dim))
        idx += n
    draws_all = np.concatenate(draws_list, axis=1)
    print("All draws shape:", draws_all.shape)
