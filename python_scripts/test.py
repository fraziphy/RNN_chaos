import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'key1' : [7, 2.72, 2+3j],
            'key2' : ( 'abc', 'xyz')}
    
    data1 = np.array([1,2,6,9,10])
else:
    data = None
    data1 = None
data = comm.bcast(data, root=0)
data1 = comm.scatter(data1, root=0)
print(f"the rank is {rank} and {data1} and {data}")
