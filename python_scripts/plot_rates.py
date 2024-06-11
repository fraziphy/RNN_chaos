import numpy as np
import matplotlib.pyplot as plt
import pickle
import os



# Set model parameters from environment variables
session = int(os.environ.get("SESSION"))


with open(f'./data/data_synaptic_upscaling_chaos_session_{session}.pickle', 'rb') as handle:
    data = pickle.load(handle)
    
    
# Simulation parameters
T, dt = 7000, 1
size = 200
p = 0.1
rep = 20

# Define synaptic upscaling levels
synaptic_upscaling = np.arange(1,24,3)

colors=["b","r"]
for g in synaptic_upscaling:
    for j in range(rep):
        
        fig, axs = plt.subplots(nrows=20, ncols=10, figsize=(30, 60))
        axs = axs.ravel()
        for c, rate in enumerate(["rates1", "rates2"]):
        
            
            a = data["g={}".format(g)][rate]["rep={}".format(j)]
            
            
            for i in range(size):
                axs[i].plot(a[i], c=colors[c])
                
        for ax in axs.flat:
            ax.set_xticks([0, 2000, 4000, 6000])
            ax.set_xticklabels([0, 2, 4, 6])
            
        fig.suptitle(f"session_{session}_g_{g}_rates1_blue_rates2_red_rep_{j}",fontsize=30)
        fig.savefig(f"./figures/fig_session_{session}_g_{g}_rep_{j}.png")
        plt.close()
