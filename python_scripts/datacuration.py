import numpy as np
import pickle

# Simulation parameters
T, dt = 7000, 1
size = 200
p = 0.1
reps = 20


def Fit(y_aux):
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


data_to_save = {}

Lyapunov_exponent_all = np.zeros((5,5),dtype=float)

for session in [1, 2, 3, 4, 5]:
    data_to_save[f"session={session}"] = {}
    

    data = None
    with open(f'./data/data_synaptic_upscaling_chaos_session_{session}.pickle', 'rb') as handle:
        data = pickle.load(handle)
        
    
    Lyapunov_exponent = np.array([])
        
    for i, g in enumerate(synaptic_upscaling):
        
        data_to_save[f"session={session}"][f"g={g}"] = {}
        
        rep = 4
        lim = 100
        if g==4:
            rep = 16
            lim = 500
        if g==7:
            rep = 13
            lim = 4000
        if g==10:
            rep = 6
            lim = 4000
        if g==13:
            rep = 5
            lim = 4000
        if g==16:
            rep = 15
            lim = 4000
        if g==19:
            rep = 5
            lim = 4000
        if g==22:
            rep = 13
            lim = 4000
        
        
        data_to_save[f"session={session}"][f"g={g}"]["example"] = data[f"g={g}"]["rates1"][f'rep={rep}'][:,:lim]
            
        if session==1 and g==10:
            data_to_save[f"session={session}"][f"g={g}"]["example_2"] = data[f"g={g}"]["rates1"][f'rep={13}'][:,:lim]
        
        
        diff = np.zeros((reps, int(T/dt)), dtype=float)
        m, n = diff.shape

        for ii in range(reps):
            rates1 = data[f"g={g}"]["rates1"][f'rep={ii}']
            rates2 = data[f"g={g}"]["rates2"][f'rep={ii}']
            for j in range(n):
                diff[ii, j] = np.sqrt(np.dot(rates2[:, j] - rates1[:, j], rates2[:, j] - rates1[:, j]))
        
        
        if session == 1 and g == 10:
            
            stables = []
            for ii in range(20):
                if diff[ii,-1] < 1:
                    stables.append(ii)

            for ii in range(20):
                if diff[ii,-1] >= 1:
                    stables.append(ii)
            
            data_to_save[f"session={session}"][f"g={g}"]["delta"] = diff[stables]
            data_to_save[f"session={session}"][f"g={g}"]["delta_index"] = stables
        else:
            data_to_save[f"session={session}"][f"g={g}"]["delta"] = diff
            
        
        
        
        if g>=10:
            
            non_stables = []
            for j in range(reps):
                if diff[j,-1] >= 1:
                    non_stables.append(j)
                    
            point = np.where(np.log(diff[non_stables].mean(axis=0)) >= 0.8 * np.log(diff[non_stables].mean(axis=0)).max())[0][0]
            Lyapunov_exponent_aux = Fit(np.log(diff[non_stables].mean(axis=0))[:point])
            
            if g ==13:
                # Create a polynomial function from the coefficients: Lyapunov_exponent_aux
                polynomial = np.poly1d(Lyapunov_exponent_aux)
                # Use the fitted line to predict y-values
                x_fit = np.arange(len(np.log(diff[non_stables].mean(axis=0))[:point]))
                y_fit = polynomial(x_fit)
                
                data_to_save[f"session={session}"][f"g={g}"]["delta_x_fit"] = x_fit
                data_to_save[f"session={session}"][f"g={g}"]["delta_y_fit"] = y_fit
                
            
            data_to_save[f"session={session}"][f"g={g}"]["delta_mean"] = diff[non_stables].mean(axis=0)
            data_to_save[f"session={session}"][f"g={g}"]["delta_point"] = point
            
            
            
        
            Lyapunov_exponent = np.append(Lyapunov_exponent, Lyapunov_exponent_aux[0])
    
    Lyapunov_exponent_all[session-1] = Lyapunov_exponent
    
data_to_save["Lyapunov_exponent"] = Lyapunov_exponent_all
    
    
with open('./data/data_curation.pickle', 'wb') as handle:
    # Use pickle.dump to serialize the DATA object and write it to the file
    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
