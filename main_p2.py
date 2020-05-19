import rosenbluth2 as r
import numpy as np
import pickle
import h5py

## simulation parameters
eps, T, sigma, res, = 1, 1, 0.8, 6
parameters = (T,eps,sigma,res)
## defines bead length for simulations as range(start,bb,step) 
bb= 100
start = 1
step = 5

################################
#### Rosenbluth simulation  ####
################################

# runs defines the number of polymers grown per bead length
runs = 100
bb += 2
polymer_list = list()

for count, N in enumerate(range(start,bb,step)):
    polymer_arr = list()

    for i in range(runs):
        polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
        polymer_arr.append(list(r.genPoly(polymer,N,parameters)))
        
    polymer_list.append(polymer_arr)
    print('Done with {} beads'.format(N))
  
# Optional saving and loading of data in pickle format  
#pickle.dump(polymer_list, open( "Sim_rosenbluth_{}.p".format(str(runs)), "wb" ))   
#polymer_list = pickle.load( open( "Sim_rosenbluth_{}.p".format(str(runs)), "rb" ) )

A_w_RR= np.zeros(len(range(start,bb,step)))
A_w_err_RR = np.zeros(len(range(start,bb,step)))

r_gyration_RR = np.zeros(len(range(start,bb,step)))
r_gyration_err_RR = np.zeros(len(range(start,bb,step)))
#
for count, i in enumerate(polymer_list):
    # extract end-to-end distance
    S, S_err = r.getS(i)
    A_w_RR[count] = S
    A_w_err_RR[count] = S_err
    
    #extract radius of gyration
    Rg, Rg_err = r.getRg(i)
    r_gyration_RR[count] = Rg
    r_gyration_err_RR[count] = Rg_err

################################
######## PERM simulation  ######
################################

initial_pop_size = 100

A_w_PERM = np.zeros(len(range(start,bb,step)))
A_w_err_PERM = np.zeros(len(range(start,bb,step)))

r_gyration_PERM = np.zeros(len(range(start,bb,step)))
r_gyration_err_PERM = np.zeros(len(range(start,bb,step)))

pop_size_info = np.zeros(shape=(len(range(start,bb,step))))

for count, b in enumerate(range(start,bb,step)):

    step_size = 1 # bead added aech time before pruning/enrichment
    num_steps = b # number of times we prune/enrich
    # final number of beads
    # N = 3 + step_size*num_steps
    
    population = r.initializePopulation(initial_pop_size,parameters)
    for i in range(num_steps):
        population = r.grow(population, step_size, 2.4, 0.3, parameters)
    
    pop_size_info[count] = len(population)
    
    # extract end-to-end distance
    S, S_err = r.getS(population)
    A_w_PERM[count] = S
    A_w_err_PERM[count] = S_err
    
    #extract radius of gyration
    Rg, Rg_err = r.getRg(population)
    r_gyration_PERM[count] = Rg
    r_gyration_err_PERM[count] = Rg_err


    print('Done with {} beads'.format(b))
    
## optional saving of data in h5py format
#hf = h5py.File('Sim_polymer_PERM.h5', 'w')
#hf.create_dataset('A_w', data=A_w_PERM)
#hf.create_dataset('A_w_err', data=A_w_err_PERM)
#
#hf.create_dataset('r_gyration', data=r_gyration_PERM)
#hf.create_dataset('r_gyration_err', data=r_gyration_err_PERM)
#
#hf.create_dataset('pop_size_info', data=pop_size_info)
#hf.close()
    


##############################
###### Plot generation  ######
##############################
# every dsteps'th data point is shown
dstep = 2
# plot end-to-end distance with data of Rosenbluth and PERM obtain fit parameters
param_Rn_RR, param_Rn_PERM = r.fitFunc(A_w_RR[:], A_w_err_RR[:], A_w_PERM, A_w_err_PERM, start, bb, step, dstep, pop_size_info, plot=True)

# plot end-to-end distance with data of Rosenbluth and PERM obtain fit parameters
param_Rg_RR, param_Rg_PERM = r.fitFuncRg(r_gyration_RR[:], r_gyration_err_RR[:], r_gyration_PERM, r_gyration_err_PERM, start, bb, step, dstep, plot=True)
















