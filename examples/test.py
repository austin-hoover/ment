import numpy as np
from mpi4py import MPI

print("HI")
comm = MPI.COMM_WORLD
print(comm.Get_rank())
