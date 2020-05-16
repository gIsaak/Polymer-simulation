# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:17:39 2020

@author: ludwig
"""
import rosenbluth2_server_calc as r
import numpy as np
import pickle


#server
#############################################
#### uncorrelated simulation isacco dtype ###
#############################################
bb= 251
start = 1
step = 1
runs = 10000
polymer_list = list()

for count, N in enumerate(range(start,bb,step)):
    polymer_arr = list()

    for i in range(runs):
        polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
        polymer_arr.append(list(r.genPoly(polymer,N)))
        
    polymer_list.append(polymer_arr)
    print('Done with {} beads'.format(N))
    
pickle.dump(polymer_list, open( "Sim_rosenbluth_10000.p", "wb" ))   

#load data to extract observables
#poly = pickle.load( open( "save1000.p", "rb" ) )
#A_w = np.zeros(len(range(start,bb,step)))
#A_w_err = np.zeros(len(range(start,bb,step)))
#
#r_gyration = np.zeros(len(range(start,bb,step)))
#r_gyration_err = np.zeros(len(range(start,bb,step)))
#
#for count, i in enumerate(poly):
#    # extract end-to-end distance
#    S, S_err = r.getS(i)
#    A_w[count] = S
#    A_w_err[count] = S_err
#    
#    #extract radius of gyration
#    Rg, Rg_err = r.getRg(i)
#    r_gyration[count] = Rg
#    r_gyration_err[count] = Rg_err


