#!/bin/bash
##############################################################
#                            BASH JOB                        #
#                                                            #
#                                                            #
##############################################################
# 
# 
# 
# 
SESSIONS="1 2 3 4 5"
for i in ${SESSIONS};do
    export SESSION=${i}
#     mpirun -np 8 python3 ./python_scripts/parallel_simulations.py
#     python3 ./python_scripts/plot_rates.py
done


# python3 ./python_scripts/plot_each_session.py

# datacurationJobPID=$(python3 ./python_scripts/datacuration.py)
# 
# wait $datacurationJobPID

python3 ./python_scripts/plot_main_figures.py
