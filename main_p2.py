import rosenbluth as r
import matplotlib.pyplot as plt
import numpy as np


rosenbluth_parameters = {
    
    # Simulation
    'number_of_beads':      5000,     # Number of beads that should be added
    'kb':                   1,      # Boltzman constant
    
    # Initialial Configuration
    'temperature':          300,    # Temperature of the system
    
    # Plotting options
    'plotting':             False,  # Plot the self avoiding random walk
    'plot_delay':           0.05    # Time until new bead is being added
   
    }

################################
###loop over many simulations###
################################

param = list()
beads = list()
tot_dist_list = list()
runs = 10
for i in range(runs):
    bead, tot_dist = r.rosenbluth(rosenbluth_parameters)
    tot_dist_list.append(tot_dist)
    beads.append(bead)
    param.append(r.fit_func(tot_dist))
    print('Done with {}th run'.format(i))

# after math   
summ = 0
for count, i in enumerate(param):
    if len(np.trim_zeros(tot_dist_list[count],'b')) > 1500:
        print(count)
        summ += i[1]
print('power law exponent: {}'.format(summ/runs))

################################
### single simulation, plots ###
################################

#beads, tot_dist = r.rosenbluth(rosenbluth_parameters)
#r.fit_func(tot_dist)
