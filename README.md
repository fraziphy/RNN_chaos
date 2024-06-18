# RNN_chaos

### Introduction
This GitHub repository contains scripts designed to explore the impact of synaptic upscaling on the chaotic dynamics of spontaneous neural activity within recurrent neural networks (RNNs), utilizing a rate-based neuron model.

## Repository Structure
```
heterogeneous_synaptic_homeostasis
├── data
│   └── data_curation.pickle
├── figures
│   └── results.pdf
├── python_scripts
│   ├── synaptic_upscaling_chaos_module.py
│   ├── parallel_simulations.py
│   ├── datacuration.py
│   ├── plot_final_figure.py
│   └── test.py
├── job.sh
├── bash_function.sh
├── LICENSE
└── README.md
```

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of job.sh
The **_job.sh**_ script is an executable that submits simulation tasks. To execute the **_job.sh**_ script, use the following commands in the terminal:
```
$ chmod +x job.sh
$ ./job.sh
```

Briefly, the script is organized into blocks, each corresponding to specific simulations. The first block corresponds to the simulations of five sessions, each with a unique connectivity matrix in the RNN. Within each session, a randomly generated connection matrix was used. Each session included 8 episodes, corresponding to 8 different synaptic upscaling factors. To facilitate the simulation speed, the Message Passing Interface (MPI) is utelized. In particular, mpi4py is employed to utelizes the 8 cpus, distributing the 8 episodes across 8 cpus. The second block corresponds to data curation, and finally, the third block corresponds to the plots of the results.

To summarize, the script is segmented into blocks, each allocated for specific tasks. The opening block encompasses simulations across five distinct sessions, each characterized by a unique connectivity matrix within the RNN. In each session, a connection matrix is randomly generated. These sessions are comprised of eight episodes, reflecting different levels of synaptic upscaling in the RNN. To boost simulation performance, mpi4py is employed to fully utilize all 8 CPUs, distributing the episodes equitably among them. Subsequently, there is a block dedicated to data curation, and the concluding block is tasked with creating visual representations of the outcomes.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of python scripts
The Python scripts are located within the _python_scripts_ directory and include:
- _synaptic_upscaling_chaos_module.py_ 
- _parallel_simulations.py_
- _datacuration.py_
- _plot_final_figure.py_ 
- _test.py_

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_synaptic_upscaling_chaos_module.py_:** This script is designed to model the behavior of a Recurrent Rate Neural Network (RRNN), featuring $`N=200`$ rate neurons. It incorporates a sparse connectivity matrix, where each element is assigned a value of 0 with a probability of $`1 − p`$, where the sparsity parameter $`p=0.1`$. To ensure network complexity and diversity, self-connections within the network are explicitly set to zero.

The synaptic strengths, represented by non-zero elements in the connectivity matrix, are generated through independent sampling (across sessions) from Gaussian distributions. These distributions are characterized by a mean of zero and variances that are inversely proportional to the product of $`N ∗ p`$, ensuring a balanced and dynamic network structure. Additionally, the synaptic strengths are scaled by a factor of $`g`$ (the synaptic upscaling factor). This scaling mechanism introduces variability in the network's connectivity, allowing for the investigation of how different synaptic upscaling values influence the RRNN's dynamics and functionality. By adjusting the synaptic upscaling factor, we can explore the impact of varying synaptic strengths on the network's performance. This experimentation provides valuable insights into the RRNN's capabilities and constraints, offering a deeper understanding of its potential applications and limitations.

**Python requirements:**
- numpy

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_parallel_simulations.py_:** This script is designed to numerically integrate a Recurrent Neural Network (RNN) under various configurations, focusing on the impact of a synaptic upscaling factor ($`g`$) on network dynamics. The script utilizes a connectivity matrix and an environment variable named SESSION to manage distinct configurations of the connectivity matrix across different sessions.

- _SESSION variable_: Utilizes the SESSION environment variable to manage different connectivity matrices, allowing for comprehensive analysis across multiple configurations.
- _Episodic Trials_: Each session consists of 8 episodes, with the synaptic upscaling factor ($`g`$) varying from 1 to 21 increments of 3 across episodes. This design enables a thorough examination of how different synaptic upscaling values affect the network's behavior.
- _Network Initialization_: Each episode contains 20 trials, with each trial initializing the network with a unique configuration. The firing rate vector (r) is initialized with uniform random values between $`\[-1, 1\]`$, and the network simulation runs for 7 seconds per trial.
- _Chaotic Activity Evaluation_: To assess the presence of chaotic activity, the script employs the Lyapunov exponent, a metric that measures the system's sensitivity to initial conditions. Variations of $`10^{-4}`$ are introduced in the initial conditions of each neuron across trials within episodes and sessions to observe the divergence of trajectories over time, indicative of chaotic behavior.
- _Parallel Processing_: Leverages the mpi4py library to distribute the computation of 8 different synaptic upscalings across 8 CPUs, optimizing computational efficiency and scalability.

**Python requirements:**
- _synaptic_upscaling_chaos_module.py_
- numpy
- mpi4py
- pickle
- os

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_datacuration.py_:** This script is designed for data curation and analysis, specifically targeting the computation of the Lyapunov exponent for RNNs. It stores the spontaneous dynamics of a single neuron across different synaptic upscaling factors and the initial condition variations, providing insights into the network's behavior under various conditions.

- _Dynamic Analysis Across Sessions_: Captures the network's spontaneous dynamics for a single neuron in an example session, considering different synaptic upscaling factors and slight variations in itial conditions (by $`10^{-4}`$).
- _Lyapunov Exponent Calculation_: Assesses the Lyapunov exponent across various synaptic upscaling factors for the dynamics of neighboring trajectories in the phase space, offering a quantitative measure of chaos in the system.
- _Trajectory Divergence Analysis_: Computes the divergence of neighboring trajectories over time, leveraging the following formula to quantify the level of chaos:
```math
||\delta(t)|| = ||\delta(0)|| exp(\lambda t)
```
Here, $`\lambda`$ represents the Lyapunov exponent. The difference at each time point is determined using:
```math
||\delta(t)|| = ||r^{+} (t) − r(t)||
```
Where $`r^{+} (t)`$ denotes the network state at time $`t`$ with altered initial conditions by $`10^{-4}`$. The Lyapunov exponent is extracted from the slope of $`\ln ||\delta(t)||`$ before reaching the saturation point, achieving 80% of its maximum value.

**Python requirements:**
- numpy
- pickle

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_plot_final_figure_:** This script is aimed at plotting the results. It also visualizes the graph representation of an exemplary RNN. Leveraging the powerful NetworkX library in Python, it provides a straightforward way to generate and display the structural of the experimental design. It creates a graphical representation of an RNN, showcasing the connections and nodes, and how synaptic upscaling affects the network structure in an intuitive format.

**Python requirements:**
- numpy
- pickle
- matplotlib
- seaborn
- networkx
- cmasher

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_test.py_:** This script is designed to assess the feasibility and effectiveness of conducting parallel simulations using the MPI.

**Python requirements:**
- numpy
- mpi4py

to run the **_test.py_** script, propmt the following command from the repository directory:

```
$ mpirun -np 8 --oversubscribe python3 ./python_scripts/test.py
```


------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of data
This directory includes a python dictionary in the format of pickle containing processed data (_data_curation.pickle_).

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of figures

This directory contains a PDF file as results.pdf showcasing the results of simulations

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contributing

Thank you for considering contributing to our project! We welcome contributions from the community to help improve our project and make it even better. To ensure a smooth contribution process, please follow these guidelines:

1. **Fork the Repository**: Fork our repository to your GitHub account and clone it to your local machine.

2. **Branching Strategy**: Create a new branch for your contribution. Use a descriptive branch name that reflects the purpose of your changes.

3. **Code Style**: Follow our coding standards and style guidelines. Make sure your code adheres to the existing conventions to maintain consistency across the project.

4. **Pull Request Process**:
    Before starting work, check the issue tracker to see if your contribution aligns with any existing issues or feature requests.
    Create a new branch for your contribution and make your changes.
    Commit your changes with clear and descriptive messages explaining the purpose of each commit.
    Once you are ready to submit your changes, push your branch to your forked repository.
    Submit a pull request to the main repository's develop branch. Provide a detailed description of your changes and reference any relevant issues or pull requests.

5. **Code Review**: Expect feedback and review from our maintainers or contributors. Address any comments or suggestions provided during the review process.

6. **Testing**: Ensure that your contribution is properly tested. Write unit tests or integration tests as necessary to validate your changes. Make sure all tests pass before submitting your pull request.

7. **Documentation**: Update the project's documentation to reflect your changes. Include any necessary documentation updates, such as code comments, README modifications, or user guides.

8. **License Agreement**: By contributing to our project, you agree to license your contributions under the terms of the project's license (GNU General Public License v3.0).

9. **Be Respectful**: Respect the opinions and efforts of other contributors. Maintain a positive and collaborative attitude throughout the contribution process.

We appreciate your contributions and look forward to working with you to improve our project! If you have any questions or need further assistance, please don't hesitate to reach out to us.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Credits

- **Author:** [Farhad Razi](https://github.com/fraziphy)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contact

- **Contact information:** [email](farhad.razi.1988@gmail.com)
