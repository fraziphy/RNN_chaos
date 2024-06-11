import numpy as np
import matplotlib.pyplot as plt
import pickle





with open('./data/data_curation.pickle', 'rb') as handle:
    data = pickle.load(handle)




session = 1







# Create a figure and a set of subplots with specified dimensions
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

# Adjust the horizontal space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.7)

# Plotting on each subplot

for j, g in enumerate([1, 4, 7]):
    rep = 4
    if g==4:
        rep = 16
    if g==7:
        rep = 13
    
    

    
    id = 161
    if g==4:
        id = 73
    if g==7:
        id = 32
    axs[j, 0].plot(data[f"session={session}"][f"g={g}"]["example"][id], color='#7f7f7f')
    axs[j, 0].set_title('One Representative Neuron')
    axs[j, 0].set_ylabel('Firing Rate (au)')

    lim = 100
    if g==4:
        lim = 500
    if g==7:
        lim = 2000
        
    g1 = axs[j, 1].imshow(data[f"session={session}"][f"g={g}"]["example"][:,:lim], cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
    axs[j, 1].invert_yaxis()
    cbar = plt.colorbar(g1)
    cbar.set_label('Firing Rate (au)')
    axs[j, 1].spines["top"].set_visible(False)
    axs[j, 1].spines["bottom"].set_visible(False)
    axs[j, 1].spines["right"].set_visible(False)
    axs[j, 1].spines["left"].set_visible(False)
    cbar.outline.set_visible(False)
    cbar.set_ticks([-1,0,1])

    axs[j, 1].set_yticks(np.array([0, 99, 199]))
    axs[j, 1].set_yticklabels(np.array([0, 99, 199])+1)
    
    xticks = np.array([0, 99]) 
    xticklabels = np.array([0, 0.1]) 
    if g==4:
        xticks = np.array([0, 499]) 
        xticklabels = np.array([0, 0.5]) 
    if g==7:
        xticks = np.array([0,999, 1999])
        xticklabels = np.array([0, 1, 2])
    
    axs[j, 1].set_xticks(xticks+ 0.5)
    axs[j, 1].set_xticklabels(xticklabels, rotation=0)
    
    axs[j, 1].annotate('', xy=(0, id), xycoords='data',
                xytext=(-lim/8, id), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(arrowstyle='->, head_width=0.4',color='#7f7f7f',lw=2.5))
    axs[j, 1].set_title('The Representative Trial')

    axs[j, 1].set_ylabel("Neuron id")




    lim = 100
    if g==4:
        lim = 200
    if g==7:
        lim = 3000

    g2 = axs[j, 2].imshow(data[f"session={session}"][f"g={g}"]["delta"][:,:lim], cmap="copper", vmin=0, aspect='auto')
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
    cbar.set_ticks([0, 0.0005, 0.001])
    if g==7:
        cbar.set_ticks([0, 0.005, 0.01])
    

    axs[j, 2].set_yticks(np.array([0, 9, 19]))
    axs[j, 2].set_yticklabels(np.array([0, 9, 19])+1)
    
    xticks = np.array([0, 99]) 
    xticklabels = np.array([0, 0.1]) 
    if g==4:
        xticks = np.array([0, 199]) 
        xticklabels = np.array([0, 0.2]) 
    if g==7:
        xticks = np.array([0,999, 1999, 2999])
        xticklabels = np.array([0, 1, 2, 3])
    
    axs[j, 2].set_xticks(xticks+ 0.5)
    axs[j, 2].set_xticklabels(xticklabels, rotation=0)
    
    axs[j, 2].annotate('', xy=(0, rep), xycoords='data',
                xytext=(-lim/8, rep), textcoords='data',
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
fig.savefig("./figures/fig1.pdf")
plt.close()














# Create a figure and a set of subplots with specified dimensions
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))

# Adjust the horizontal space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.7)

# Plotting on each subplot
g = 10

for j, id in enumerate([161, 42]):
    if id ==161:
        axs[j, 0].plot(data[f"session={session}"][f"g={g}"]["example"][id], color='#7f7f7f')
    else:
        axs[j, 0].plot(data[f"session={session}"][f"g={g}"]["example_2"][id], color='#7f7f7f')
        
    axs[j, 0].set_xlim(0,4000)
    axs[j, 0].set_xticks([0, 2000, 4000])
    axs[j, 0].set_xticklabels([0, 2, 4])
    axs[j, 0].set_title('One Representative Neuron')
    axs[j, 0].set_ylabel('Firing Rate (au)')
    
    if id ==161:
        g1 = axs[j, 1].imshow(data[f"session={session}"][f"g={g}"]["example"][:,:4000], cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
    else:
        g1 = axs[j, 1].imshow(data[f"session={session}"][f"g={g}"]["example_2"][:,:4000], cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')        
    
    axs[j, 1].invert_yaxis()
    cbar = plt.colorbar(g1)
    cbar.set_label('Firing Rate (au)')
    axs[j, 1].spines["top"].set_visible(False)
    axs[j, 1].spines["bottom"].set_visible(False)
    axs[j, 1].spines["right"].set_visible(False)
    axs[j, 1].spines["left"].set_visible(False)
    cbar.outline.set_visible(False)
    cbar.set_ticks([-1,0,1])
    
    axs[j, 1].set_yticks(np.array([0, 99, 199]))
    axs[j, 1].set_yticklabels(np.array([0, 99, 199])+1)
    
    axs[j, 1].set_xticks(np.array([0,1999, 3999]))
    axs[j, 1].set_xticklabels(np.array([0,2, 4]), rotation=0)
    
    axs[j, 1].set_ylabel("Neuron id")
    
    
    axs[j, 1].annotate('', xy=(0, id), xycoords='data',
                xytext=(-500, id), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(arrowstyle='->, head_width=0.4',color='#7f7f7f',lw=2.5))
    axs[j, 1].set_title('The Representative Trial')
    


j=0
g2 = axs[j, 2].imshow(data[f"session={session}"][f"g={g}"]["delta"], cmap="copper", aspect='auto')
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



axs[j, 2].set_yticks(np.array([0, 9, 19]))
axs[j, 2].set_yticklabels(np.array([0, 9, 19])+1)

xticks = np.array([0, 1999, 3999, 5999]) 
xticklabels = np.array([0, 2, 4, 6]) 

axs[j, 2].set_xticks(xticks)
axs[j, 2].set_xticklabels(xticklabels, rotation=0)

axs[j, 2].annotate('', xy=(0, 3), xycoords='data',
            xytext=(-7000/8, 3), textcoords='data',
            va='top', ha='left',
            arrowprops=dict(arrowstyle='->, head_width=0.4',color='k',lw=2.5))

axs[j, 2].annotate('', xy=(0, 15), xycoords='data',
            xytext=(-7000/8, 15), textcoords='data',
            va='top', ha='left',
            arrowprops=dict(arrowstyle='->, head_width=0.4',color='k',lw=2.5))


axs[j, 2].set_title('Sensitivity')

axs[j, 2].set_ylabel("Trial id")

# Optionally, add labels and titles to each subplot
for ax in axs.flat:
    ax.set(xlabel='Time (s)')


axs[0,0].text(-0.2,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=axs[0,0].transAxes)
axs[0,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[0,1].transAxes)


axs[1,0].text(-0.2,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axs[1,0].transAxes)
axs[1,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[1,1].transAxes)

axs[0,2].text(-0.2,1.15,r"$\bf{c}$",horizontalalignment='left', transform=axs[0,2].transAxes)

axs[1,2].remove()

# Save the plot
fig.savefig("./figures/fig2.pdf")
plt.close()


















# Create a figure and a set of subplots with specified dimensions
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 14))

# Adjust the horizontal space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.7)

# Plotting on each subplot
ids = [36, 145, 40, 167]
reps = [4,12,5,15]
for j, g in enumerate([13, 16, 19, 22]):
    
    
    rep = 4
    if g==4:
        rep = 16
    if g==7:
        rep = 13
    if g==10:
        rep = 6
    if g==13:
        rep = 5
    if g==16:
        rep = 15
    if g==19:
        rep = 5
    if g==22:
        rep = 13
    
    

    id = ids[j]
    
    axs[j, 0].plot(data[f"session={session}"][f"g={g}"]["example"][id], color='#7f7f7f')
    axs[j, 0].set_title('One Representative Neuron')
    axs[j, 0].set_xlim(0,4000)
    axs[j, 0].set_xticks([0, 2000, 4000])
    axs[j, 0].set_xticklabels([0, 2, 4])
    axs[j, 0].set_ylabel('Firing Rate (au)')
    
    g1 = axs[j, 1].imshow(data[f"session={session}"][f"g={g}"]["example"][:,:4000], cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
    axs[j, 1].invert_yaxis()
    cbar = plt.colorbar(g1)
    cbar.set_label('Firing Rate (au)')
    axs[j, 1].spines["top"].set_visible(False)
    axs[j, 1].spines["bottom"].set_visible(False)
    axs[j, 1].spines["right"].set_visible(False)
    axs[j, 1].spines["left"].set_visible(False)
    cbar.outline.set_visible(False)
    cbar.set_ticks([-1,0,1])
    
    axs[j, 1].set_yticks(np.array([0, 99, 199]))
    axs[j, 1].set_yticklabels(np.array([0, 99, 199])+1)
    
    axs[j, 1].set_xticks(np.array([0,1999, 3999]))
    axs[j, 1].set_xticklabels(np.array([0,2, 4]), rotation=0)
    
    axs[j, 1].set_ylabel("Neuron id")
    
    axs[j, 1].annotate('', xy=(0, id), xycoords='data',
                xytext=(-500, id), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(arrowstyle='->, head_width=0.4',color='#7f7f7f',lw=2.5))
    axs[j, 1].set_title('The Representative Trial')



    lim = 7000
    g2 = axs[j, 2].imshow(data[f"session={session}"][f"g={g}"]["delta"][:,:lim], cmap="copper", vmin=0, aspect='auto')
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
    

    axs[j, 2].set_yticks(np.array([0, 9, 19]))
    axs[j, 2].set_yticklabels(np.array([0, 9, 19])+1)
    
    xticks = np.array([0, 1999, 3999, 5999]) 
    xticklabels = np.array([0, 2, 4, 6]) 
    
    axs[j, 2].set_xticks(xticks)
    axs[j, 2].set_xticklabels(xticklabels, rotation=0)
    
    axs[j, 2].annotate('', xy=(0, rep), xycoords='data',
                xytext=(-lim/8, rep), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(arrowstyle='->, head_width=0.4',color='k',lw=2.5))
    if j==0:
        axs[j, 2].annotate('', xy=(0, 11), xycoords='data',
                xytext=(-lim/14, 11), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(arrowstyle='->, head_width=0.4',color='r',lw=2.5))
    axs[j, 2].set_title('Sensitivity')

    axs[j, 2].set_ylabel("Trial id")


# Optionally, add labels and titles to each subplot
for ax in axs.flat:
    ax.set(xlabel='Time (s)')


axs[0,0].text(-0.2,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=axs[0,0].transAxes)
axs[0,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[0,1].transAxes)
axs[0,2].text(-0.2,1.15,"(iii)",horizontalalignment='left', transform=axs[0,2].transAxes)

axs[1,0].text(-0.2,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axs[1,0].transAxes)
axs[1,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[1,1].transAxes)
axs[1,2].text(-0.2,1.15,"(iii)",horizontalalignment='left', transform=axs[1,2].transAxes)

axs[2,0].text(-0.2,1.15,r"$\bf{c}$"+"(i)",horizontalalignment='left', transform=axs[2,0].transAxes)
axs[2,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[2,1].transAxes)
axs[2,2].text(-0.2,1.15,"(iii)",horizontalalignment='left', transform=axs[2,2].transAxes)

axs[3,0].text(-0.2,1.15,r"$\bf{d}$"+"(i)",horizontalalignment='left', transform=axs[3,0].transAxes)
axs[3,1].text(-0.2,1.15,"(ii)",horizontalalignment='left', transform=axs[3,1].transAxes)
axs[3,2].text(-0.2,1.15,"(iii)",horizontalalignment='left', transform=axs[3,2].transAxes)




# Save the plot
fig.savefig("./figures/fig3.pdf")
plt.close()















# Create a figure and a set of subplots with specified dimensions
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

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
    y = data[f"session={session}"][f"g={g}"]["delta_mean"]
    axs[0].plot(y, c=colors[i],label=f'g={g}')


axs[0].set_xlim(0,7000)
axs[0].set_xticks([0, 2000, 4000, 6000])
axs[0].set_xticklabels([0, 2, 4, 6])
axs[0].set_yticks([0, 5, 10, 15])
axs[0].set_ylabel(r'$||\delta ||$')
axs[0].set_xlabel('Time (s)')
axs[0].legend()



for i, g in enumerate([10,13,16,19,22]):
    y = data[f"session={session}"][f"g={g}"]["delta_mean"]
    axs[1].plot(np.log(y), c=colors[i],label=f'g={g}')
    point = data[f"session={session}"][f"g={g}"]["delta_point"]
    axs[1].plot(point, np.log(y)[point],"ro")



g=13
x_fit = data[f"session={session}"][f"g={g}"]["delta_x_fit"]
y_fit = data[f"session={session}"][f"g={g}"]["delta_y_fit"]

axs[1].plot(x_fit,y_fit,"k")

axs[1].legend()




g2 = axs[2].imshow(data["Lyapunov_exponent"], cmap="binary", vmin=0, aspect='auto')
axs[2].invert_yaxis()
cbar = plt.colorbar(g2)
cbar.set_label("Lyapunov Exponent")
axs[2].spines["top"].set_visible(False)
axs[2].spines["bottom"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].spines["left"].set_visible(False)
cbar.outline.set_visible(False)
cbar.formatter.set_powerlimits((0, 0))

# to get 10^3 instead of 1e3
cbar.formatter.set_useMathText(True)

yticks = np.arange(5)
yticklabels = np.arange(1,6)
axs[2].set_yticks(yticks)
axs[2].set_yticklabels(yticklabels)

xticks = np.arange(5)
xticklabels = np.arange(10,24,3)
axs[2].set_xticks(xticks)
axs[2].set_xticklabels(xticklabels, rotation=0)

axs[2].annotate('', xy=(-2, 0), xycoords='data',
            xytext=(-0, 0), textcoords='data',
            va='top', ha='left',
            arrowprops=dict(arrowstyle='->, head_width=0.4',color='k',lw=2.5))

axs[2].set_ylabel("Session id")
axs[2].set_xlabel("Synaptic Upscaling Factor (g)")


axs[1].set_xlim(0,4000)
axs[1].set_xticks([0, 2000, 4000])
axs[1].set_xticklabels([0, 2, 4])
axs[1].set_ylabel(r'$\ln ||\delta ||$')
axs[1].set_xlabel('Time (s)')




axs[0].text(-0.2,1.05,r"$\bf{a}$",horizontalalignment='left', transform=axs[0].transAxes)

axs[1].text(-0.2,1.05,r"$\bf{b}$",horizontalalignment='left', transform=axs[1].transAxes)

axs[2].text(-0.2,1.05,r"$\bf{C}$",horizontalalignment='left', transform=axs[2].transAxes)


#Save the plot
fig.savefig("./figures/fig4.pdf")
plt.close()
