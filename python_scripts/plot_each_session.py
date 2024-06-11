import numpy as np
import matplotlib.pyplot as plt
import pickle



def Fit(y_aux):
    x_aux = np.arange(len(y_aux))

    # Fit a line to the data
    coefficients = np.polyfit(x_aux, y_aux, 1)
    
    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)
    
    # Use the fitted line to predict y-values
    y_fit = polynomial(x_aux)

    return coefficients[0], coefficients[1]




T, dt = 7000, 1
size = 200
p = 0.1
reps = 20

for session in [1, 2, 3, 4, 5]:
    
    with open(f'./data/data_synaptic_upscaling_chaos_session_{session}.pickle', 'rb') as handle:
        data = pickle.load(handle)



    data_to_save = {}

    for g in range(1,24,3):
        
        diff = np.zeros((reps, int(T/dt)), dtype=float)
        m, n = diff.shape

        for i in range(reps):
            rates1 = data[f"g={g}"]["rates1"][f'rep={i}']
            rates2 = data[f"g={g}"]["rates2"][f'rep={i}']
            for j in range(n):
                diff[i, j] = np.sqrt(np.dot(rates2[:, j] - rates1[:, j], rates2[:, j] - rates1[:, j]))
        data_to_save[f"g={g}"] = diff
        
        
    slopes = np.zeros(5,dtype=float)
    for i,g in enumerate(np.arange(10,24,3)):
        point = np.where(np.log(data_to_save[f"g={g}"].mean(axis=0)) >= 0.8 * np.log(data_to_save[f"g={g}"].mean(axis=0)).max())[0][0]
        slopes[i] = Fit(np.log(data_to_save[f"g={g}"].mean(axis=0))[:point])[0]
        
    
    synaptic_upscaling = np.arange(1,24,3)

    # Create a figure and a set of subplots with specified dimensions
    fig, axs = plt.subplots(nrows=len(synaptic_upscaling), ncols=3, figsize=(15, 3.5*len(synaptic_upscaling)))

    # Adjust the horizontal space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.7)

    # Plotting on each subplot

    for j, g in enumerate(synaptic_upscaling):
        rep = 4
        id = 161
        if g==4:
            rep = 16
            id = 73
        if g==7:
            rep = 13
            id = 32
        if g==10:
            rep = 16
            id = 73
        if g==13:
            rep = 5
            id = 165
        if g==16:
            rep = 15
            id = 50
        if g==19:
            rep = 5
            id = 173
        if g==22:
            rep = 13
            id = 70
        axs[j, 0].plot(data[f"g={g}"]["rates1"][f'rep={rep}'][id], color='#7f7f7f')
        axs[j, 0].set_title('One Representative Neurons')
        axs[j, 0].set_ylabel('Firing Rate (au)')

        lim = 100
        if g==4:
            lim = 500
        if g>=7:
            lim = 4000
            
        g1 = axs[j, 1].imshow(data[f"g={g}"]["rates1"][f'rep={rep}'][:,:lim], cmap="bwr", vmin=-1, vmax=1, aspect='auto')
        axs[j, 1].invert_yaxis()
        cbar = plt.colorbar(g1)
        cbar.set_label('Firing Rate (au)')
        axs[j, 1].spines["top"].set_visible(False)
        axs[j, 1].spines["bottom"].set_visible(False)
        axs[j, 1].spines["right"].set_visible(False)
        axs[j, 1].spines["left"].set_visible(False)
        cbar.outline.set_visible(False)
        cbar.set_ticks([-1,0,1])

        axs[j, 1].set_yticks(np.array([0, 99, 199]) + 0.5)
        axs[j, 1].set_yticklabels(np.array([0, 99, 199])+1)
        
        xticks = np.array([0, 99]) 
        xticklabels = np.array([0, 0.1]) 
        if g==4:
            xticks = np.array([0, 499]) 
            xticklabels = np.array([0, 0.5]) 
        if g>=7:
            xticks = np.array([0, 1999, 3999])
            xticklabels = np.array([0, 2, 4])
        
        axs[j, 1].set_xticks(xticks+ 0.5)
        axs[j, 1].set_xticklabels(xticklabels, rotation=0)
        
        axs[j, 1].annotate('', xy=(0, id+.5), xycoords='data',
                    xytext=(-lim/8, id+.5), textcoords='data',
                    va='top', ha='left',
                    arrowprops=dict(arrowstyle='->, head_width=0.4',color='#7f7f7f',lw=2.5))
        axs[j, 1].set_title('The Representative Trial')

        axs[j, 1].set_ylabel("Neuron id")




        lim = 100
        if g==4:
            lim = 500
        if g>=7:
            lim = 7000
        
        stables = []
        for ij in range(20):
            if data_to_save[f"g={g}"][ij,-1] < 1:
                stables.append(ij)
        
        for ij in range(20):
            if data_to_save[f"g={g}"][ij,-1] >= 1:
                stables.append(ij)
        
        g2 = axs[j, 2].imshow(data_to_save[f"g={g}"][stables,:lim], cmap="copper", vmin=0, aspect='auto')
        axs[j, 2].invert_yaxis()
        cbar = plt.colorbar(g2)
        cbar.set_label(r'$||\delta||$')
        axs[j, 2].spines["top"].set_visible(False)
        axs[j, 2].spines["bottom"].set_visible(False)
        axs[j, 2].spines["right"].set_visible(False)
        axs[j, 2].spines["left"].set_visible(False)
        cbar.outline.set_visible(False)
        cbar.formatter.set_powerlimits((0, 0))

        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)
        if g<7:
            cbar.set_ticks([0, 0.0005, 0.001])
        

        axs[j, 2].set_yticks(np.array([0, 9, 19]))
        axs[j, 2].set_yticklabels(np.array([0, 9, 19])+1)
        
        xticks = np.array([0, 99]) 
        xticklabels = np.array([0, 0.1]) 
        if g==4:
            xticks = np.array([0, 499]) 
            xticklabels = np.array([0, 0.5]) 

        if g>=7:
            xticks = np.array([0, 1999, 3999, 5999])
            xticklabels = np.array([0, 2, 4, 6])
        
        axs[j, 2].set_xticks(xticks+ 0.5)
        axs[j, 2].set_xticklabels(xticklabels, rotation=0)
        
        axs[j, 2].annotate('', xy=(0, rep+.5), xycoords='data',
                    xytext=(-lim/8, rep+.5), textcoords='data',
                    va='top', ha='left',
                    arrowprops=dict(arrowstyle='->, head_width=0.4',color='k',lw=2.5))
        axs[j, 2].set_title('Sensitivity')

        axs[j, 2].set_ylabel("Trial id")



    axs[0, 0].set_xlim(0,100)
    axs[0, 0].set_xticks([0, 100])
    axs[0, 0].set_xticklabels([0, 0.1])

    axs[1, 0].set_xlim(0,500)
    axs[1, 0].set_xticks([0, 500])
    axs[1, 0].set_xticklabels([0, 0.5])

    axs[2, 0].set_xlim(0,2000)
    axs[2, 0].set_xticks([0, 1000, 2000])
    axs[2, 0].set_xticklabels([0, 1, 2])

    for i in range(2, len(synaptic_upscaling)):
        axs[i, 0].set_xlim(0,4000)
        axs[i, 0].set_xticks([0, 2000, 4000])
        axs[i, 0].set_xticklabels([0, 2, 4])
        


    # Optionally, add labels and titles to each subplot
    for ax in axs.flat:
        ax.set(xlabel='Time (s)')

    axs[0,0].text(-0.2,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=axs[0,0].transAxes)
    axs[0,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[0,1].transAxes)
    axs[0,2].text(-0.2,1.15,"(iii)",horizontalalignment='left', transform=axs[0,2].transAxes)

    axs[1,0].text(-0.2,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axs[1,0].transAxes)
    axs[1,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[1,1].transAxes)
    axs[1,2].text(-0.2,1.15,"(iii)",horizontalalignment='left', transform=axs[1,2].transAxes)

    axs[2,0].text(-0.2,1.15,r"$\bf{C}$"+"(i)",horizontalalignment='left', transform=axs[2,0].transAxes)
    axs[2,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[2,1].transAxes)
    axs[2,2].text(-0.2,1.15,"(iii)",horizontalalignment='left', transform=axs[2,2].transAxes)


    fig.align_ylabels(axs[:,0])


    # Save the plot
    fig.savefig(f"./figures/fig_session_{session}_first.pdf")
    plt.close()
    
    
    
    
    
    
    
    # Create a figure and a set of subplots with specified dimensions
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 2.5))

    # Adjust the horizontal space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.7)

    # Plotting on each subplot


    colors = [
        
        '#ff7f0e', # Orange
        '#9467bd', # Purple
        '#8c564b', # Brown
        '#2ca02c', # Green
        '#e377c2', # Pink
        '#98FB98', # Light Green
        '#7f7f7f', # Gray
        '#1f77b4', # Blue
        '#d62728' # Red
    ]


    for i, g in enumerate([10,13,16,19,22]):
        if g in [10,13]:
            non_stables = []
            for j in range(20):
                if data_to_save[f"g={g}"][j,-1] >= 1:
                    non_stables.append(j)
            axs[0].plot(data_to_save[f"g={g}"][non_stables].mean(axis=0), c=colors[i],label=f'g={g}')

        else:
            axs[0].plot(data_to_save[f"g={g}"].mean(axis=0), c=colors[i],label=f'g={g}')

    axs[0].set_xlim(0,7000)
    axs[0].set_xticks([0, 2000, 4000, 6000])
    axs[0].set_xticklabels([0, 2, 4, 6])
    axs[0].set_yticks([0, 5, 10, 15])
    axs[0].set_ylabel(r'$||\delta ||$')
    axs[0].set_xlabel('Time (s)')
    axs[0].legend()



    for i, g in enumerate([10,13,16,19,22]):
        non_stables = []
        for j in range(20):
            if data_to_save[f"g={g}"][j,-1] >= 1:
                non_stables.append(j)
        axs[1].plot(np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0)), c=colors[i],label=f'g={g}')
        point = np.where(np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0)) >= 0.8 * np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0)).max())[0][0]
        axs[1].plot(point, np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0))[point],"ro")



    g=13
    non_stables = []
    for j in range(20):
        if data_to_save[f"g={g}"][j,-1] >= 1:
            non_stables.append(j)
    point = np.where(np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0)) >= 0.8 * np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0)).max())[0][0]
    coefficients = Fit(np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0))[:point])
    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)
    # Use the fitted line to predict y-values
    y_fit = polynomial(np.arange(len(np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0))[:point])))

    axs[1].plot(np.arange(len(np.log(data_to_save[f"g={g}"][non_stables].mean(axis=0))[:point])),y_fit,"k")

    axs[1].legend()



    axs[2].plot(np.arange(10,24,3),slopes,"ko")
    axs[2].set_xticks(np.arange(10,24,3))
    axs[2].set_xlabel("Synaptic Upscaling Factor")
    axs[2].set_ylabel("Lyapunov Exponent")


    axs[1].set_xlim(0,4000)
    axs[1].set_xticks([0, 2000, 4000])
    axs[1].set_xticklabels([0, 2, 4])
    # axs[1].set_yticks([0, 5, 10, 15])
    axs[1].set_ylabel(r'$\ln ||\delta ||$')
    axs[1].set_xlabel('Time (s)')




    axs[0].text(-0.2,1.15,r"$\bf{a}$",horizontalalignment='left', transform=axs[0].transAxes)

    axs[1].text(-0.2,1.15,r"$\bf{b}$",horizontalalignment='left', transform=axs[1].transAxes)

    axs[2].text(-0.2,1.15,r"$\bf{C}$",horizontalalignment='left', transform=axs[2].transAxes)


    # Save the plot
    fig.savefig(f"./figures/fig_session_{session}_second.pdf")
    plt.close()
    
    
