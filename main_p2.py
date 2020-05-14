import rosenbluth2 as r
import matplotlib.pyplot as plt
import numpy as np
import h5py


#################################
#### single simulation, plots ###
#################################
#
#
#polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
#
#polymer, polWeight = r.genPoly(polymer,10)
#
#fig = plt.figure()
#plot = plt.plot(polymer[:,0],polymer[:,1], '-bo')
#
#plt.show()

#################################
#### uncorrelated simulation  ###
#################################
bb= 51
start = 10
step = 1
A_w = np.zeros(len(range(start,bb,step)))
A_w_err = np.zeros(len(range(start,bb,step)))
polWeight_list = list()
r_gyration = np.zeros(len(range(start,bb,step)))
r_gyration_err = np.zeros(len(range(start,bb,step)))
r_gyration_list = list()
for count, b in enumerate(range(start,bb,step)):
    runs = 100
    polWeight_arr = np.zeros(runs)
    end_dist_arr = np.zeros(runs)
    r_gyration_arr = np.zeros(runs)
#    r_gyration_arr_err = np.zeros(runs)
    for i in range(runs):
        
        polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
        polymer, polWeight  = r.genPoly(polymer,b)
        
        polWeight_arr[i] = polWeight
        end_dist_arr[i] = r.getTotDist(polymer)
        
        #radius of gyration
        x_c = np.mean(polymer[:,0])
#        x_c_err = np.std(polymer[:,0])
        y_c = np.mean(polymer[:,1])
#        y_c_err = np.std(polymer[:,1])
        r_mean = np.sqrt(x_c**2+y_c**2)
#        r_mean_err = np.sqrt(x_c_err**2+y_c_err**2)
        r_i = np.sqrt(np.sum(polymer**2,axis=1))
        R = np.sum((r_i-r_mean)**2)/len(polymer)
#        R_err = np.sum((r_mean_err)**2)/len(polymer)
        r_gyration_arr[i] = R
#        r_gyration_arr_err[i] = R_err
    # end-to-end dist weighted avg
    A_w[count] = np.sum(polWeight_arr[:]*end_dist_arr[:])/np.sum(polWeight_arr[:])
    A_w_err[count] =  np.sum(polWeight_arr[:]*(end_dist_arr[:])**2)/np.sum(polWeight_arr[:]) - A_w[count]**2
    polWeight_list.append(polWeight_arr)
    
    r_gyration[count] = np.sum(polWeight_arr[:]*r_gyration_arr[:])/np.sum(polWeight_arr[:])
    r_gyration_err[count] = np.sum(polWeight_arr[:]*(r_gyration_arr[:])**2)/np.sum(polWeight_arr[:]) - r_gyration[count]**2
    r_gyration_list.append(r_gyration_arr) 
    print('Done with {} beads'.format(b))

#param = r.fitFunc(A_w[:],A_w_err,plot=True)
    
# optional saving of data in binary file
hf = h5py.File('Sim_polymer.h5', 'w')
hf.create_dataset('A_w', data=A_w)
hf.create_dataset('A_w_err', data=A_w_err)
hf.create_dataset('polWeight_list', data=polWeight_list)
hf.create_dataset('r_gyration', data=r_gyration)
hf.create_dataset('r_gyration_err', data=r_gyration_err)
hf.create_dataset('r_gyration_list', data=r_gyration_list)
hf.close()
