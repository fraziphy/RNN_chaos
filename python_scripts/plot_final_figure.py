import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import findfont, FontProperties
import matplotlib.patches as patches

import cmasher as cmr






params = {'legend.fontsize': 7,
         'axes.labelsize': 7,
         'axes.titlesize':7,
         'xtick.labelsize':7,
         'ytick.labelsize':7,
         'axes.linewidth': 0.4,
         'xtick.major.width':0.4,
         'ytick.major.width':0.4,
         'xtick.major.size':3,
         'ytick.major.size':3}
plt.rcParams.update(params)


plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
# 
# 
# 
# 
font = findfont(FontProperties(family=['sans-serif']))



colors = ['gold', 'cyan', 'magenta', 'red', 'green', 'blue',  'darkorange', 'purple']


with open('./data/data_curation.pickle', 'rb') as handle:
    data = pickle.load(handle)
    




def draw_brace(ax, xspan, yy):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy - (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(y, x, color='black', lw=3)





def plot_network(ax,base, scale,edge_list,g):
        
    G = nx.DiGraph()
    
    G.add_weighted_edges_from(edge_list)
    pos = {}
    pos_array = np.array([[-0.7, -0.9], [0.1, -1.1], 
                          [-0.1, -0.4], [1, 0.8], 
                          [-0.2, 0.8], [0.8, -0.7], 
                          [-1, -0.3], [-1, 0.4], 
                          [0.1, 0.4], [1.1, 0.]])
    pos_array[:,0] *= scale[0]
    pos_array[:,1] *= scale[1]
    pos_array[:,0] += base[0]
    pos_array[:,1] += base[1]
    for i in range(10):
        pos[i+1] = pos_array[i]
    
    
    
    nx.draw_networkx_nodes(G, pos, node_size=100, ax=ax,node_color="gray")
    # nx.draw_networkx_nodes(G, pos, node_size=100, ax=ax)
    # nx.draw_networkx_labels(G, pos, ax=ax)
    
    #fig.savefig("1.png", bbox_inches='tight', pad_inches=0)
    
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    
    curved_edges_weight = [G.get_edge_data(edge[0],edge[1])["weight"] for edge in curved_edges]
    straight_edges_weight = [G.get_edge_data(edge[0],edge[1])["weight"] for edge in straight_edges]
    
    curved_edges_exc = [curved_edges[i] for i in range(len(curved_edges)) if curved_edges_weight[i]>0]
    curved_edges_inh = [curved_edges[i] for i in range(len(curved_edges)) if curved_edges_weight[i]<0]
    
    
    edges_weight = straight_edges_weight + curved_edges_weight
    
    # Convert the list of integers to a color map
    vmin_aux = min(edges_weight)
    vmax_aux = max(edges_weight)
    vmax = max(vmax_aux,-vmin_aux)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    colors = cmr.iceburn(norm(edges_weight)) # Use viridis colormap as an example
    
    arcs1 = nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges,arrowstyle='-|>, head_length=.6, head_width=0.2',width=g)
    offset = 0
    for i, arc in enumerate(arcs1):  # change alpha values of arcs
        arc.set_color(colors[i])
        if edges_weight[i]<0:
            arc.set_arrowstyle('-[, widthB=0.4, lengthB=.15')
        offset += 1
    
    
    
    arc_rad = 0.25
    arcs2 = nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges,arrowstyle='-|>, head_length=.6, head_width=0.2', connectionstyle=f'arc3, rad = {arc_rad}',width=g)
    
    for i, arc in enumerate(arcs2):  # change alpha values of arcs
        arc.set_color(colors[offset+i])
        if edges_weight[offset+i]<0:
            arc.set_arrowstyle('-[, widthB=0.4, lengthB=.15')
    
    # sm = plt.cm.ScalarMappable(cmap=cmr.gem, norm=Normalize(vmin=-vmax, vmax=vmax))
    # sm.set_array([]) # Set the array to empty since we're not mapping it to anything
    
    # # Now explicitly specify the ax for the colorbar
    # cb = fig.colorbar(sm, ax=ax)
    # cb.set_ticks([-vmax,0,vmax])
    # for t in cb.ax.get_yticklabels():
    #     t.set_horizontalalignment('right')   
    #     t.set_x(5.9)
    #     t.set_fontsize(15)
    
    # ax.axis("off")
    
    
    
    
    
    
def Fig1(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.34#of the axis x lenght
    ws_y = 0.25
    distx = 0.5 #of the axis x lenght
    disty = 2
    fix_x_aux = fix_x-left-right
    ax_x = (fix_x_aux)/(2+1*distx)
    ax_y = ax_x/8
    
    fix_y = (6+5*ws_y+disty+8 + 7*ws_y) * ax_y
    fix_y_aux = fix_y-top-bottom
    ax_y = fix_y_aux/(6+5*ws_y+disty+8 + 7*ws_y)
    
    
    gsEXP = GridSpec(1,1, bottom=bottom+(disty*.7+8 + 7*ws_y)*ax_y/fix_y_aux, top=-top+(6+5*ws_y+disty+8 + 7*ws_y)*ax_y/fix_y_aux, left=0.2*left, right=0.8*(-right)+1, wspace=ws_x,hspace = ws_y)
    gsEXP_1 = GridSpec(1,1, bottom=bottom+0.125+(disty*.7+8 + 7*ws_y)*ax_y/fix_y_aux, top=-top-0.02+(6+5*ws_y+disty+8 + 7*ws_y)*ax_y/fix_y_aux, left=0.875, right=0.912, wspace=ws_x,hspace = ws_y)
    gs1 = GridSpec(8,1, bottom=bottom, top=bottom+(8 + 7*ws_y)*ax_y/fix_y_aux, left=left, right=left+(1)*ax_x/fix_x_aux, wspace=ws_x,hspace = ws_y)
    gs1_aux1 = GridSpec(1,1, bottom=bottom, top=bottom+(8 + 7*ws_y)*ax_y/fix_y_aux, left=left, right=left+(1.05)*ax_x/fix_x_aux, wspace=ws_x,hspace = ws_y)
    gs1_aux2 = GridSpec(1,1, bottom=bottom, top=bottom+(8 + 7*ws_y)*ax_y/fix_y_aux, left=left-0.03, right=left+(1)*ax_x/fix_x_aux, wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(1,1, bottom=bottom+( (8 + 7*ws_y)/2 + disty/2)*ax_y/fix_y_aux, top=bottom+(8 + 7*ws_y)*ax_y/fix_y_aux, left=left+(1+distx)*ax_x/fix_x_aux, right=-right+(2+1*distx)*ax_x/fix_x_aux, wspace=ws_x,hspace = ws_y)
    gs3 = GridSpec(1,1, bottom=bottom, top=bottom+( (8 + 7*ws_y)/2 - disty/2)*ax_y/fix_y_aux, left=left+(1+distx)*ax_x/fix_x_aux, right=-right+(2+1*distx)*ax_x/fix_x_aux, wspace=ws_x,hspace = ws_y)
    
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axEXP = []
    axEXP.append(fig.add_subplot(gsEXP[0]))
    axEXP_1 = []
    axEXP_1.append(fig.add_subplot(gsEXP_1[0]))
    axA_aux = []
    axA_aux.append(fig.add_subplot(gs1_aux1[0]))
    axA_aux.append(fig.add_subplot(gs1_aux2[0]))
    axA = []
    for i in range(8):
        axA.append(fig.add_subplot(gs1[i]))
    axB = []
    axB.append(fig.add_subplot(gs2[0]))
    axC = []
    axC.append(fig.add_subplot(gs3[0]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Define the rectangles' properties
    rectangle_width = 30
    rectangle_height = 15
    # Calculate actual overlap
    overlapx = 0.05 * rectangle_width
    overlapy = 0.03* rectangle_height
    
    origins =[(0,0)]
    for i in range(1,5):
        a = origins[0][0]+i*overlapx
        b = origins[0][1]+i*overlapy
        origins.append((a,b))
    
    # Draw each rectangle with smooth corners and different colors
    for i, origin in enumerate(origins):
        patch = patches.Rectangle(origin, rectangle_width, rectangle_height, edgecolor="gray", facecolor="lightgray")
        axEXP[0].add_patch(patch)
    
    
        
        
    
    
    centers = [(56,7.5)] # centers for each Ellipse
    # Define the Ellipse' properties
    Ellipse_width = 25
    Ellipse_height = 15
    # Calculate actual overlap
    # overlapx = 0.05 * Ellipse_width
    overlapy = 0.018 * Ellipse_height
    for i in range(1,8):
        a = centers[0][0]+i*overlapy
        b = centers[0][1]+i*overlapy
        centers.append((a,b))
    
    # Draw each Ellipse with different centers
    for i, center in enumerate(centers):
        circle = patches.Ellipse(center, width=Ellipse_width, height=Ellipse_height, edgecolor="gray", facecolor="lightgray") # Centered at (0, 0)
        axEXP[0].add_patch(circle)
    
    
    
        
    
    
    # Define the rectangles' properties
    rectangle_width = 14
    rectangle_height = 10.5
    # Calculate actual overlap
    overlapx = 0.028 * rectangle_width
    overlapy = 0.03 * rectangle_height
    
    origins =[(78,0)]
    for i in range(1,20):
        a = origins[0][0]+i*overlapx
        b = origins[0][1]+i*overlapy
        origins.append((a,b))
    
    # Draw each rectangle with smooth corners and different colors
    for i, origin in enumerate(origins):
        patch = patches.Rectangle(origin, rectangle_width, rectangle_height, edgecolor="gray", facecolor="lightgray")
        axEXP[0].add_patch(patch)
    
    
    
        
    
    
    edge_list_1 = [(1,2,-0.1),(2,1,0.17),(7,1,-0.08), (1,3,0.11), (7,3,-0.18),
                     (2,6,-0.08),(10,9,-0.15),(3,6,0.02),(6,10,-0.09),(9,10,0.1),
                     (3,9,0.12),(8,9,-0.11),(9,6,-0.17),(4,10,0.18),(4,5,0.02),
                     (8,5,-0.02),(9,7,0.06),(9,5,-0.2)]
    
    
    plot_network(axEXP[0],(25.5,8.3),(8,4.6),edge_list_1,1)
    plot_network(axEXP[0],(58,8.3),(8,4.6),edge_list_1,2.5)
    off_x = 22
    off_y = 14
    axEXP[0].annotate("", xy=(1.4+off_x, off_y+1.), xytext=(1.4+off_x-4, off_y+1.), transform=axEXP[0].transAxes,arrowprops=dict(arrowstyle='-[, widthB=0.4, lengthB=.15', lw=1,color='gray'), zorder=100)
    axEXP[0].annotate("", xy=(10+off_x+1., off_y+1.), xytext=(10+off_x-4, off_y+1.), transform=axEXP[0].transAxes,arrowprops=dict(arrowstyle='-|>, head_length=0.6, head_width=0.2', lw=1,color='gray'), zorder=100)
    off_x1 = 0.23
    off_y1 = 0.87
    axEXP[0].text(0.015+off_x1,off_y1,"INH", transform=axEXP[0].transAxes, fontsize=7)
    axEXP[0].text(0.106+off_x1,off_y1,"EXC", transform=axEXP[0].transAxes, fontsize=7)
    
    center = (8,15.)
    neuron_width = 1.92
    neuron_height = 1.25
    circle = patches.Ellipse(center, width=neuron_width, height=neuron_height, edgecolor="gray", facecolor="gray") # Centered at (0, 0)
    axEXP[0].add_patch(circle)
    off_x1 = 0.
    off_y1 = 0.87
    axEXP[0].text(0.103+off_x1,off_y1,"Rate Neuron", transform=axEXP[0].transAxes, fontsize=7)
    
    
    
    
    
    
    off_x = 56
    off_y = 14.7
    axEXP[0].annotate("", xy=(off_x, off_y+1.), xytext=(off_x-4, off_y+1.), transform=axEXP[0].transAxes,arrowprops=dict(arrowstyle='-[, widthB=0.4, lengthB=.15', lw=2.5,color='gray'), zorder=100)
    axEXP[0].annotate("", xy=(7+off_x+1., off_y+1.), xytext=(7+off_x-4, off_y+1.), transform=axEXP[0].transAxes,arrowprops=dict(arrowstyle='-|>, head_length=0.6, head_width=0.2', lw=2.5,color='gray'), zorder=100)
    off_y = 13.2
    axEXP[0].annotate("", xy=(off_x, off_y+1.), xytext=(off_x-4, off_y+1.), transform=axEXP[0].transAxes,arrowprops=dict(arrowstyle='-[, widthB=0.4, lengthB=.15', lw=1,color='gray'), zorder=100)
    axEXP[0].annotate("", xy=(7+off_x+1., off_y+1.), xytext=(7+off_x-4, off_y+1.), transform=axEXP[0].transAxes,arrowprops=dict(arrowstyle='-|>, head_length=0.6, head_width=0.2', lw=1,color='gray'), zorder=100)
    
    off_x1 = 0.535
    off_y1 = 0.87
    axEXP[0].text(off_x1,off_y1,"||", transform=axEXP[0].transAxes, fontsize=7)
    axEXP[0].text(0.07+off_x1,off_y1,"||", transform=axEXP[0].transAxes, fontsize=7)
    
    off_x1 = 0.505
    off_y1 = 0.833
    axEXP[0].text(off_x1,off_y1,"g x", transform=axEXP[0].transAxes, fontsize=7)
    axEXP[0].text(0.07+off_x1,off_y1,"g x", transform=axEXP[0].transAxes, fontsize=7)
    
    
    
    
    
    
    sm = plt.cm.ScalarMappable(cmap=cmr.iceburn, norm=Normalize(vmin=-0.2, vmax=0.2))
    sm.set_array([]) # Set the array to empty since we're not mapping it to anything
    cax = fig.add_axes([0.14,0.7,0.006,0.13])
    cbar = fig.colorbar(sm, cax=cax)
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-0.2,0,0.2])
    cbar.set_label('Synaptic Strength', labelpad=0)
    cax.yaxis.set_label_position('left')
    
    
    
    
    axEXP[0].axis("off")
    axEXP[0].set_xlim(-1, 100)
    axEXP[0].set_ylim(-1, 17)
    axEXP[0].set_title("Experimental Design\n\n", weight='bold')
    axEXP[0].text(0.2,1.01,"Five sessions, each\nwith a unique connectivity matrix",horizontalalignment='center', transform=axEXP[0].transAxes, fontsize=7)
    axEXP[0].text(0.58,1.01,"One session contains eight episodes, each\nwith a unique synaptic upscaling factor g",horizontalalignment='center', transform=axEXP[0].transAxes, fontsize=7)
    axEXP[0].text(0.88,1.01,"One episode contains twenty trials, each\nwith a unique network initialization",horizontalalignment='center', transform=axEXP[0].transAxes, fontsize=7)
    
    
    
    start_point_main = (35.5, 8.5) # Starting point of the main arrow
    end_point_main = (41, 8.5)    # Ending point of the main arrow
    axEXP[0].annotate("", xy=end_point_main, xytext=start_point_main,
                arrowprops=dict(arrowstyle="-",facecolor='black', lw=3),
                )
    draw_brace(axEXP[0], (1, 16), 42)
    
    
    start_point_main = (69.5, 8.5) # Starting point of the main arrow
    end_point_main = (75, 8.5)    # Ending point of the main arrow
    axEXP[0].annotate("", xy=end_point_main, xytext=start_point_main,
                arrowprops=dict(arrowstyle="-",facecolor='black', lw=3),
                )
    draw_brace(axEXP[0], (1, 16), 76)
    
    
    
    a_aux = 2 * np.random.random(200) - 1
    g2 = axEXP_1[0].imshow(a_aux.reshape(-1,1), cmap="coolwarm", vmin = -1, vmax = 1, aspect='auto')
    axEXP_1[0].invert_yaxis()
    cbar = plt.colorbar(g2)
    cbar.set_label("Firing rate (au)")
    axEXP_1[0].spines["top"].set_visible(False)
    axEXP_1[0].spines["bottom"].set_visible(False)
    axEXP_1[0].spines["right"].set_visible(False)
    axEXP_1[0].spines["left"].set_visible(False)
    cbar.outline.set_visible(False)
    cbar.set_ticks([-1,0,1])
    for t in cbar.ax.get_yticklabels():
        t.set_horizontalalignment('right')   
        t.set_x(2.9)
    
    axEXP_1[0].set_yticks(np.array([0, 199]))
    axEXP_1[0].set_yticklabels(np.array([0, 199])+1)
    
    axEXP_1[0].set_xticks([])
    
    axEXP_1[0].set_ylabel("Neuron id", labelpad=-10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    synaptic_upscaling = np.arange(1,23,3)
    
    
    
    
    axA_aux[0].spines["top"].set_visible(False)
    axA_aux[0].spines["left"].set_visible(False)
    axA_aux[0].spines["bottom"].set_visible(False)
    axA_aux[0].set_xticks([])
    axA_aux[0].yaxis.tick_right()
    axA_aux[0].yaxis.set_label_position("right")
    axA_aux[0].set_ylabel("Synaptic Upscaling Factor (g)")
    x = 1/(8+7*ws_y)
    axA_aux[0].set_yticks(np.linspace(x/2,x/2+(7+7*ws_y)*x,8))
    axA_aux[0].set_yticklabels(synaptic_upscaling[::-1])
    
    axA_aux[0].set_ylim(0,1)
    
    
    
    axA_aux[1].spines["top"].set_visible(False)
    axA_aux[1].spines["left"].set_visible(False)
    axA_aux[1].spines["right"].set_visible(False)
    axA_aux[1].spines["bottom"].set_visible(False)
    axA_aux[1].set_xticks([])
    axA_aux[1].set_yticks([])
    axA_aux[1].set_ylabel("Firing Rate (au)")
    
    
    
    
    for i, g in enumerate(synaptic_upscaling):
        axA[i].plot(data["instance"][f"g={g}"]["rates1"],c=colors[i],lw=1,zorder=0)
        axA[i].plot(data["instance"][f"g={g}"]["rates2"],"--",c=colors[i],lw=1,zorder=1)
        axA[i].set_xlim(0,2000)
        axA[i].set_ylim(-1.05,1.05)
    
        axA[i].spines["top"].set_visible(False)
        axA[i].spines["right"].set_visible(False)
    
        
        axA[i].set_xticks([0,1000,2000])
        axA[i].set_xticklabels([0,1,2])
        if i!=7:
            
            axA[i].spines["bottom"].set_visible(False)
            axA[i].set_xticks([])
    axA[-1].set_xlabel("Time (s)")
    axA[0].set_title("Spontaneous Activity of a Typical Neuron", weight='bold')
    
    
    for i, g in enumerate(synaptic_upscaling):
        if g > 7:
            axB[0].plot(data["session=1"][f"g={g}"]["delta_mean"],c=colors[i],lw=1,label=f"g={g}")
    axB[0].set_xlim(0,7000)
    axB[0].set_xticks([0, 2000, 4000, 6000])
    axB[0].set_xticklabels([0, 2, 4, 6])
    axB[0].set_yticks([0, 5, 10, 15])
    axB[0].set_xlabel("Time (s)")
    axB[0].set_ylabel(r'$||\delta ||$')
    axB[0].set_title("Mean Divergance Magnitude Across Trials", weight='bold')
    axB[0].legend(fontsize=7)
    
    
    
    g2 = axC[0].imshow(data["Lyapunov_exponent"], cmap="binary", vmin=0, vmax=data["Lyapunov_exponent"].max(), aspect='auto')
    axC[0].invert_yaxis()
    cbar = plt.colorbar(g2)
    cbar.set_label("Lyapunov Exponent (s" + r"$^{-1}$" +")")
    axC[0].spines["top"].set_visible(False)
    axC[0].spines["bottom"].set_visible(False)
    axC[0].spines["right"].set_visible(False)
    axC[0].spines["left"].set_visible(False)
    cbar.outline.set_visible(False)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_ticks([0,7,14])
    
    # to get 10^3 instead of 1e3
    cbar.formatter.set_useMathText(True)
    
    yticks = np.arange(5)
    yticklabels = np.arange(1,6)
    axC[0].set_yticks(yticks)
    axC[0].set_yticklabels(yticklabels)
    
    xticks = np.arange(5)
    xticklabels = np.arange(10,24,3)
    axC[0].set_xticks(xticks)
    axC[0].set_xticklabels(xticklabels, rotation=0)
    
    axC[0].annotate('', xy=(-2, 0), xycoords='data',
                xytext=(-0, 0), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(arrowstyle='->, head_width=0.4',color='k',lw=2.5))
    
    axC[0].set_ylabel("Session id")
    axC[0].set_xlabel("Synaptic Upscaling Factor (g)")
    axC[0].set_title("Chaotic Dynamics", weight='bold')
    
    
    for i in range(8):
        axA[i].yaxis.set_ticks([-1,0,1])
        axA[i].yaxis.set_ticklabels(axA[i].get_yticks(), va='center', ha="right")
    
    
    
    
    axEXP[0].text(0,0.99,r"$\bf{a}$",horizontalalignment='left', transform=fig.transFigure, fontsize=7)
    axA[0].text(0,0.6,r"$\bf{b}$",horizontalalignment='left', transform=fig.transFigure, fontsize=7)
    axB[0].text(0.6,0.60,r"$\bf{c}$",horizontalalignment='left', transform=fig.transFigure, fontsize=7)
    axC[0].text(0.6,0.3,r"$\bf{d}$",horizontalalignment='left', transform=fig.transFigure, fontsize=7)

    
    
    
    fig.align_ylabels([axB[0],axC[0]])
    
    fig.savefig("./figures/results.pdf",dpi=600)
    
    
    
    
left=0.065
right=0.04
top=0.065
bottom=0.055

Fig1(left,right,top,bottom)
