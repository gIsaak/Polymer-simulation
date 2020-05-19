import rosenbluth as r
import numpy as np

################################
#### Simulation parameters  ####
################################
eps, T, sigma, res, = 1, 1, 0.8, 6
parameters = (T,eps,sigma,res)
## defines bead length for simulations as range(start,bb,step)
bb = 100
start = 1
step = 5

################################
#### Rosenbluth simulation  ####
################################
runs = 100 # runs defines the number of polymers grown per bead length
bb += 2
polymer_list = list()
for count, N in enumerate(range(start,bb,step)):
    polymer_arr = list()
    for i in range(runs):
        polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
        polymer_arr.append(list(r.genPoly(polymer,N,parameters)))
    polymer_list.append(polymer_arr)
    print('Done with {} beads'.format(N))
# Calculation of physical quantities
A_w_RR= np.zeros(len(range(start,bb,step)))
A_w_err_RR = np.zeros(len(range(start,bb,step)))
r_gyration_RR = np.zeros(len(range(start,bb,step)))
r_gyration_err_RR = np.zeros(len(range(start,bb,step)))
for count, i in enumerate(polymer_list):
    S, S_err = r.getS(i) # extract end-to-end distance
    A_w_RR[count] = S
    A_w_err_RR[count] = S_err
    Rg, Rg_err = r.getRg(i) #extract radius of gyration
    r_gyration_RR[count] = Rg
    r_gyration_err_RR[count] = Rg_err

################################
######## PERM simulation  ######
################################
initial_pop_size = 100
up_lim = 2.4 #enrich
low_lim = 0.3 # prune

A_w_PERM = np.zeros(len(range(start,bb,step)))
A_w_err_PERM = np.zeros(len(range(start,bb,step)))
r_gyration_PERM = np.zeros(len(range(start,bb,step)))
r_gyration_err_PERM = np.zeros(len(range(start,bb,step)))
pop_size_info = np.zeros(shape=(len(range(start,bb,step))))
for count, b in enumerate(range(start,bb,step)):
    step_size = 1 # beads added each time before pruning/enrichment
    num_steps = b # number of times we prune/enrich
    # final number of beads: N = 3 + step_size*num_steps
    population = r.initializePopulation(initial_pop_size,parameters)
    for i in range(num_steps):
        population = r.grow(population, step_size, up_lim, low_lim, parameters)
    pop_size_info[count] = len(population)
    # Calculation of physical quantities
    S, S_err = r.getS(population) # extract end-to-end distance
    A_w_PERM[count] = S
    A_w_err_PERM[count] = S_err
    Rg, Rg_err = r.getRg(population) #extract radius of gyration
    r_gyration_PERM[count] = Rg
    r_gyration_err_PERM[count] = Rg_err
    print('Done with {} beads'.format(b))

##############################
###### Plot generation  ######
##############################
dstep = 2 # every dsteps'th data point is shown
param_Rn_RR, param_Rn_PERM = r.fitFunc(A_w_RR[:], A_w_err_RR[:], A_w_PERM, A_w_err_PERM, start, bb, step, dstep, pop_size_info, plot=True)
param_Rg_RR, param_Rg_PERM = r.fitFuncRg(r_gyration_RR[:], r_gyration_err_RR[:], r_gyration_PERM, r_gyration_err_PERM, start, bb, step, dstep, plot=True)
