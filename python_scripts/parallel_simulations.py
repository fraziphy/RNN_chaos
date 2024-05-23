# Import the NumPy library for numerical operations
import numpy as np

# Import the custom module for synaptic upscaling chaos analysis
import synaptic_upscaling_chaos_module as SUCM

# Import the pickle module for serializing and deserializing Python objects
import pickle

# Import the time module for time-related functions
import time

# Import the MPI module for parallel computing
from mpi4py import MPI

# Import the os module for environmental variables
import os




# Define the directory
DIR = os.environ.get("DIRECTORY") + "/data/"

# Set model parameters from environment variables
MPF.beta_ampa_intra = float(os.environ.get("Beta_Intra"))




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
    # Generate a connectivity matrix for the network
    conn = SUCM.Connectivity_Strength(size, p, 1/(size*p))
    np.save("connectivity_strength{}.npy".format(g),conn)

    # Initialize random initial states for the RNNs
    init_states = 2 * np.random.random((rep, size)) - 1
else:
    conn = None
    init_states = None
conn = comm.bcast(conn, root=0)
init_states = comm.bcast(init_states, root=0)







# Define synaptic upscaling levels
synaptic_upscaling = np.arange(6, 200, 4) / 10
synaptic_upscaling = np.array([0.5,1,5,10])
synaptic_upscaling = np.array([5,8,15,20])
synaptic_upscaling = np.array([6,7,9,11])
synaptic_upscaling = np.arange(1,24,3)






# Dictionary to store simulation results
data = {}

# Print a message to the terminal
print(f"Starting the script in the CPU with the rank {rank}")
g = synaptic_upscaling[rank]
print("")
print(f"    synaptic upscaling value is {g}")
print("")


# Preallocate space for storing differences in firing rates
diff = np.zeros((rep, int(T/dt)), dtype=float)
m, n = diff.shape

# Loop over repetitions
for i in range(rep):
    # Print a message to the terminal
    print(f"        repetition number {i}")

    data["rep={}".format(i)] = {}

    # Create and simulate the first RNN instance
    rnn1 = SUCM.RNN(size, g*conn, init_states[i])
    rates1 = rnn1.dynamics(T, dt)

    # Create and simulate the second RNN instance with a slightly different initial condition
    rnn2 = SUCM.RNN(size, g*conn, init_states[i] + 0.0001)
    rates2 = rnn2.dynamics(T, dt)

    # Store the firing rates of both RNN instances
    data["rep={}".format(i)]["rates1"] = rates1[:5]
    data["rep={}".format(i)]["rates2"] = rates2[:5]
    
    np.save("rates1_g_{}_rep_{}.npy".format(g,i),rates1)
    np.save("rates2_g_{}_rep_{}.npy".format(g,i),rates2)

    # Calculate the Euclidean distance between the firing rates of the two RNN instances
    for j in range(n):
        diff[i, j] = np.sqrt(np.dot(rates2[:, j] - rates1[:, j], rates2[:, j] - rates1[:, j]))

# Store the differences in firing rates for each synaptic upscaling level
data["delta_mean"] = diff.mean(axis=0)
data["delta_std"] = diff.std(axis=0)

# Save data as numpy array
np.save("diff_g_{}.npy".format(g),diff)

# Print a message to the terminal
print("")

data = comm.gather(data, root=0)

if rank == 0:
    data_to_save = {}

    for i, g in enumerate(synaptic_upscaling):
        data_to_save["g={}".format(g)] = data[i]

    # Open the file in write-binary mode ('wb') and save on the local machine
    with open('data_synaptic_upscaling_chaos.pickle', 'wb') as handle:
        # Use pickle.dump to serialize the DATA object and write it to the file
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Print a message to the terminal
    print("Script completed.")

else:
    assert data is None

# Finalize the MPI environment
MPI.Finalize()
