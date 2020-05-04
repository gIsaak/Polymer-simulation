import rosenbluth2 as r
import matplotlib.pyplot as plt
import numpy as np


rosenbluth_parameters = {
    
    # Simulation
    'number_of_beads':      250,    # Number of beads that should be added
    'kb':                   1,      # Boltzman constant
    'resolution':           50,      # resolution of new possible positions
    
    # Initialial Configuration
    'temperature':          250,    # Temperature of the system
    'sigma':                0.8,    # valley od Lennard Jones pot
    'epsilon':              1,      # epsilon in Lennard Jones pot
    
#    # Plotting options
#    'plotting':             False,  # Plot the self avoiding random walk
#    'plot_delay':           0.05    # Time until new bead is being added
   
    }



################################
###loop over many simulations###
################################

fit_param = list()
#beads = list()
tot_polWeight_list = list()
tot_dist_list = list()
runs = 1000
for i in range(runs):
    polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
    polymer, polWeight, tot_dist, tot_polWeight  = r.genPoly(polymer,rosenbluth_parameters)
    tot_dist_list.append(tot_dist)
    
#    beads.append(polymer)
    tot_polWeight_list.append(tot_polWeight)
#    print('Done with {}th run'.format(i+1))

## after math

# simple mean  
A_mean = np.mean(np.asarray(tot_dist_list),axis=0)
A_std = np.std(np.asarray(tot_dist_list),axis=0)
param = r.fit_func_std(A_mean,A_std,plot=True)
print('power law exponent mean method: {}'.format(param[1]))

# using the weights
N = rosenbluth_parameters['number_of_beads']-2
A_exp = np.zeros(N)
for i in range(N):
    a = np.asarray(tot_polWeight_list)
    b = np.asarray(tot_dist_list)
    A_exp[i] = np.sum(a[:,i]*b[:,i])/np.sum(b[:,i])
param = r.fit_func(A_exp[:150],plot=True)
print('\npower law exponent weights method: {}'.format(param[1]))

#################################
#### single simulation, plots ###
#################################
#
#
#polymer, polWeight, tot_dist = r.genPoly(polymer,rosenbluth_parameters)
#
#fig = plt.figure()
#plot = plt.plot(polymer[:,0],polymer[:,1], '-bo')
#
#plt.show()
