# Import the NumPy library for numerical operations
import numpy as np

# Import the custom module for synaptic upscaling chaos analysis
import synaptic_upscaling_chaos_module as SUCM

# Import the pickle module for serializing and deserializing Python objects
import pickle

# Import the MPI module for parallel computing
from mpi4py import MPI

# Import the os module for environmental variables
import os




# Set model parameters from environment variables
session = int(os.environ.get("SESSION"))




# Obtain the MPI communicator object
comm = MPI.COMM_WORLD

# Retrieve the total number of processes in the MPI communicator
# This indicates the total size of the distributed system
size = comm.Get_size()

# Retrieve the rank of the current process within the MPI communicator
# Each process in the communicator is assigned a unique integer rank
rank = comm.Get_rank()

# Simulation parameters
T, dt = 7000, 1
size = 200
p = 0.1
rep = 20





if rank == 0:
    
    # Random number generator
    rng = np.random.default_rng(np.random.SeedSequence(entropy=1234, spawn_key=(session,)))
    
    # Generate a connectivity matrix for the network
    conn = SUCM.Connectivity_Strength(size, p, 1/(size*p), rng)

    # Initialize random initial states for the RNNs
    init_states = 2 * rng.random((rep, size)) - 1
else:
    conn = None
    init_states = None
conn = comm.bcast(conn, root=0)
init_states = comm.bcast(init_states, root=0)







# Define synaptic upscaling levels
synaptic_upscaling = np.arange(1,24,3)


# Print a message to the terminal
print(f"Starting the script for the session {session}")
print(f"    CPU with the rank {rank}")
g = synaptic_upscaling[rank]
print("")
print(f"        synaptic upscaling value is {g}")
print("")



# Dictionary to store simulation results
data_rates1 = {}
data_rates2 = {}
data_delta = {}


# Preallocate space for storing differences in firing rates
delta = np.zeros((rep, int(T/dt)), dtype=float)
m, n = delta.shape

# Loop over repetitions
for i in range(rep):
    # Print a message to the terminal
    print(f"            repetition number {i}")

    data_rates1["rep={}".format(i)] = {}
    data_rates2["rep={}".format(i)] = {}

    # Create and simulate the first RNN instance
    rnn1 = SUCM.RNN(size, g*conn, init_states[i])
    rates1 = rnn1.dynamics(T, dt)

    # Create and simulate the second RNN instance with a slightly different initial condition
    rnn2 = SUCM.RNN(size, g*conn, init_states[i] + 0.0001)
    rates2 = rnn2.dynamics(T, dt)
    
    data_rates1["rep={}".format(i)] = rates1
    data_rates2["rep={}".format(i)] = rates2

# Print a message to the terminal
print("")

data_rates1 = comm.gather(data_rates1, root=0)
data_rates2 = comm.gather(data_rates2, root=0)

if rank == 0:
    data_to_save = {}
    
    data_to_save["connectivity_strength"] = conn

    for i, g in enumerate(synaptic_upscaling):
        data_to_save["g={}".format(g)] = {}
        data_to_save["g={}".format(g)]["rates1"] = data_rates1[i]
        data_to_save["g={}".format(g)]["rates2"] = data_rates2[i]

    # Open the file in write-binary mode ('wb') and save on the local machine
    with open(f'./data/data_synaptic_upscaling_chaos_session_{session}.pickle', 'wb') as handle:
        # Use pickle.dump to serialize the DATA object and write it to the file
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Print a message to the terminal
    print("Script completed.")

else:
    assert data_rates1 is None
    assert data_rates2 is None

# Finalize the MPI environment
MPI.Finalize()
