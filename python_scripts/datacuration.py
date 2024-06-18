# Import necessary libraries
import numpy as np
import pickle

# Simulation parameters
T, dt = 7000, 1 # Total time and time step
size = 200 # Network size (#neurons)
p = 0.1 # Probability of recurrent connections
reps = 20 # Number of repetitions for each session (network connectivity)


def Fit(y_aux):
    """
    Fits a linear model to the input data and returns the slope and intercept.
    
    Parameters:
    - y_aux: Input data array
    
    Returns:
    - Slope and intercept of the fitted line
    """
    x_aux = np.arange(len(y_aux))

    # Fit a line to the data
    coefficients = np.polyfit(x_aux, y_aux, 1)
    
    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)
    
    # Use the fitted line to predict y-values
    y_fit = polynomial(x_aux)

    return coefficients[0], coefficients[1]


# Define synaptic upscaling levels
synaptic_upscaling = np.arange(1,24,3)

# Initialize data structure for saving data
data_to_save = {}


# Initialize Lyapunov exponent matrix across sessions
Lyapunov_exponent_all = np.zeros((5,5),dtype=float)

# Loop through sessions
for session in [1, 2, 3, 4, 5]:
    
    # Load data for the current session from a pickle file
    data = None
    with open(f'./data/data_synaptic_upscaling_chaos_session_{session}.pickle', 'rb') as handle:
        data = pickle.load(handle)
        
    
    
    if session ==1:
        # Initialize data structure for saving one instance of data. # This structure will hold data for individual instances of the experiment when session number is one
        data_to_save["instance"] = {}
        
        # Iterate through each upscaling level defined earlier
        for i, g in enumerate(synaptic_upscaling):
            
            data_to_save["instance"][f"g={g}"] = {}

            data_to_save["instance"][f"g={g}"]["rates1"] = data[f"g={g}"]["rates1"][f"rep={1}"][100][:2000]
            data_to_save["instance"][f"g={g}"]["rates2"] = data[f"g={g}"]["rates2"][f"rep={1}"][100][:2000]
    
    
    
    # Initialize data structure for saving this session's data
    data_to_save[f"session={session}"] = {}
    
    # Initialize Lyapunov exponent matrix for the given session
    Lyapunov_exponent = np.array([])
    
    # Process data for each upscaling level
    for i, g in enumerate(synaptic_upscaling):
        
        data_to_save[f"session={session}"][f"g={g}"] = {}
        
        # Initialize a matrix to store differences between rates for the current upscaling level and session
        diff = np.zeros((reps, int(T/dt)), dtype=float)
        m, n = diff.shape
        
        # Iterate through each repetition
        for ii in range(reps):
            rates1 = data[f"g={g}"]["rates1"][f'rep={ii}']
            rates2 = data[f"g={g}"]["rates2"][f'rep={ii}']
            for j in range(n):
                # Calculate the Euclidean distance between two vectors representing rates at each time step
                diff[ii, j] = np.sqrt(np.dot(rates2[:, j] - rates1[:, j], rates2[:, j] - rates1[:, j]))
        
            
        
        # For upscaling levels greater than or equal to 10, calculate Lyapunov exponents
        if g>=10:
            
            # Store instances where neighboring trajectories do not converge
            non_stables = []
            for j in range(reps):
                if diff[j,-1] >= 1:
                    non_stables.append(j)
                    
            # Find the index where the log of the mean difference starts to saturate
            point = np.where(np.log(diff[non_stables].mean(axis=0)) >= 0.8 * np.log(diff[non_stables].mean(axis=0)).max())[0][0]
            # Calculate the Lyapunov exponent for the identified segment
            Lyapunov_exponent_aux = Fit(np.log(diff[non_stables].mean(axis=0))[:point])
            
            # Store the mean difference for non-stable repetition
            data_to_save[f"session={session}"][f"g={g}"]["delta_mean"] = diff[non_stables].mean(axis=0)
            
            
            
            # Append the calculated Lyapunov exponent for the given synaptic upscaling factor to the session's total Lyapunov exponents
            Lyapunov_exponent = np.append(Lyapunov_exponent, Lyapunov_exponent_aux[0])
    
    # Store the accumulated Lyapunov exponents for the current session
    Lyapunov_exponent_all[session-1] = Lyapunov_exponent
    
# After processing all sessions, convert the accumulated Lyapunov exponents to the desired units and save them
data_to_save["Lyapunov_exponent"] = 1000 * Lyapunov_exponent_all # Convert to units of 1/s instead of 1/ms
    
# Serialize and save the entire data structure to a pickle file
with open('./data/data_curation.pickle', 'wb') as handle:
    # Use pickle.dump to serialize the DATA object and write it to the file
    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
