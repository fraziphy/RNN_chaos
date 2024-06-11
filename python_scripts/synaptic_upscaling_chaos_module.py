import numpy as np
import pickle
import time




class RateNeuron:
    """
    Represents a neuron that updates its firing rate based on a total current input and a time step.
    
    Attributes:
    - threshold: Threshold for neuron activation.
    - neural_gain: Gain factor for neural response.
    - tau: Time constant for decay of synaptic currents.
    - rate: Firing rate of the neuron.
    """

    def __init__(self, init_state, tau=20.0):
        """
        Initialize the RateNeuron with specified parameters.
        
        Args:
        - init_state: Initial firing rate of the neuron.
        - threshold: Threshold for neuron activation.
        - neural_gain: Gain factor for neural response.
        - tau: Time constant for decay of synaptic currents.
        """
        self.tau = tau
        # Initialize the neuron's firing rate with the given initial state
        self.rate = init_state
    
    def update(self, total_current, dt):
        """
        Update the firing rate of the neuron based on the total current input and the time step.
        
        Args:
        - total_current: Total current input to the neuron.
        - dt: Time step for updating the neuron's state.
        
        Raises:
        - ValueError: If dt is zero, as it does not allow meaningful computation.
        """
        # Check if dt is zero and raise an exception if true
        if dt == 0:
            print("dt cannot be zero.")
            return
        
        # Compute the current based on the total current input
        current = np.tanh(total_current)
        
        # Update the neuron's firing rate based on the current and the time constant
        self.rate += (dt / self.tau) * (-self.rate + current)




class RNN:
    """
    Represents a Recurrent Neural Network (RNN) with rate-based neurons.
    
    Attributes:
    - size: Number of neurons in the network.
    - threshold: Threshold for neuron activation.
    - neural_gain: Gain factor for neural response.
    - tau: Time constant for decay of synaptic currents.
    - neurons: List of RateNeuron objects representing the neurons in the network.
    - rates: Current firing rates of the neurons.
    - recurrent_weights: Matrix of weights for recurrent connections between neurons.
    """

    def __init__(self, size, recurrent_weights, init_states, tau=20.0):
        """
        Initialize the RNN with specified parameters.
        
        Args:
        - size: Number of neurons in the network.
        - recurrent_weights: Matrix of weights for recurrent connections.
        - init_states: Initial states for each neuron.
        - threshold: Threshold for neuron activation.
        - neural_gain: Gain factor for neural response.
        - tau: Time constant for decay of synaptic currents.
        """
        self.size = size
        self.tau = tau
        # Initialize neurons with given initial states, threshold, gain, and tau
        self.neurons = [RateNeuron(init_states[i], tau) for i in range(size)]
        # Store the current firing rates of the neurons
        self.rates = [neuron.rate for neuron in self.neurons]
        # Store the recurrent weights for connections between neurons
        self.recurrent_weights = recurrent_weights
    
    def update(self, dt):
        """
        Update the state of the RNN based on the current firing rates and recurrent connections.
        
        Args:
        - dt: Time step for updating the network state.
        """
        # Compute the recurrent input currents for each neuron
        recurrent_currents = np.dot(self.recurrent_weights, self.rates)
        # Update each neuron's state based on its recurrent input current and the time step
        for i, neuron in enumerate(self.neurons):
            neuron.update(recurrent_currents[i], dt)
        # Update the current firing rates of the neurons
        self.rates = [neuron.rate for neuron in self.neurons]
    
    def dynamics(self, T, dt):
        """
        Simulate the dynamics of the RNN over a period T with a time step dt.
        
        Args:
        - T: Total duration of the simulation.
        - dt: Time step for updating the network state.
        
        Returns:
        - rate_history: A 2D array containing the firing rates of the neurons at each time step.
        """
        # Calculate the total number of iterations based on the total duration and time step
        iter = int(T/dt)
        # Initialize a 2D array to store the history of firing rates
        rate_history = np.zeros((self.size, iter), dtype=float)
        # Run the simulation
        for i in range(iter):
            # Record the current firing rates
            rate_history[:, i] = self.rates
            # Update the network state
            self.update(dt)
        # Return the history of firing rates
        return rate_history
        



def Edges(size, p, RNG):
    """
    Generates a binary adjacency matrix for a network without self-loops,
    where each connection has a probability of 1-p of being present.
    
    Args:
    - size: Number of nodes in the network.
    - p: Probability of a connection existing between any pair of nodes.
    - RNG: Random number generator
    
    Returns:
    - A boolean matrix indicating the presence of connections.
    """
    # Zero out self-connectivity
    zero_conn = np.eye(size).astype(bool)
    
    # Assign connectivities a probability of 1-p to zero
    aux = RNG.random(size**2 - size)
    zero_conn_aux = aux <= 1 - p
    
    zero_conn[~zero_conn] = zero_conn_aux
    
    return ~zero_conn


def Connectivity_Strength(size, p, std, RNG):
    """
    Constructs a weighted adjacency matrix for a network, where each connection
    has a weight drawn from a normal distribution centered around 0 with a standard
    deviation of 'std'. Connections are determined by the Edges function.
    
    Args:
    - size: Number of nodes in the network.
    - p: Probability of a connection existing between any pair of nodes.
    - std: Standard deviation of the normal distribution for connection weights.
    - RNG: Random number generator
    
    Returns:
    - A float matrix representing the connectivity strengths between nodes.
    """
    conn = np.zeros((size, size), dtype=float)
    
    # Construct a boolean matrix representing the network's connectivity
    edges = Edges(size, p, RNG)
    n_edges = len(edges[edges])
    
    # Generate connectivity strengths from a normal distribution with a mean of 0 and a standard deviation of 'std'
    aux = RNG.normal(0, std, n_edges)
    conn[edges] = aux - aux.mean()
    
    return conn
