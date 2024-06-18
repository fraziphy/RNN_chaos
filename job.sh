#!/bin/bash

# This is a header comment indicating the start of a Bash job script.

# Source a file containing custom functions used later in the script. 'bash_function.sh' contains 'process_info' that returns the CPU, MEM usuage for a given background process.
source ./bash_function.sh

# Initialize an empty string variable to hold Process IDs (PIDs).
PIDS=""

# Define session numbers as a space-separated list. These sessions will be iterated over in the following loop.
SESSIONS="1 2 3 4 5"

echo
echo "#############################"
echo "#############################"
echo "Parallel Simulations"
echo "#############################"
echo "#############################"
echo

# Loop through each session number defined above.
for i in ${SESSIONS}; do
    # Export the current session number as an environment variable.
    export SESSION=${i}
    
    # Run a Python script with MPI support, requesting 8 processes, allowing oversubscription (more processes than available cores), and capturing the PID of the last background process started.
    mpirun -np 8 --oversubscribe python3 ./python_scripts/parallel_simulations.py & PID=$!
    
    # Call a function to process information about the last background process. 'process_info' is a function defined in 'bash_function.sh' that takes a PID as an argument and returns the CPU, MEM usuage.
    process_info "$PID"
    
    # Append the current PID to the list of PIDs.
    PIDS="${PID} ${PIDS}"
done

# After all sessions have been processed, iterate over the collected PIDs.
for PID in ${PIDS}; do
    # Check if the process with the current PID is still running.
    if ps -p $PID > /dev/null; then
        # If the process is running, wait for it to finish.
        wait "$PIDS"
    fi
done

echo
echo "#############################"
echo "#############################"
echo "Data Curation"
echo "#############################"
echo "#############################"
echo

# Run another Python script for data curation in the background, capturing its PID similarly to before.
python3 ./python_scripts/datacuration.py & PID=$!

# Process information about the newly started Python script.
process_info "$PID"


# Wait for the data curation script to complete.
wait "$PID"

echo
echo "#############################"
echo "#############################"
echo "Plot Figure"
echo "#############################"
echo "#############################"
echo

# Finally, run a Python script to plot a final figure in the background, and process its information.
python3 ./python_scripts/plot_final_figure.py & PID=$!
process_info "$PID"
