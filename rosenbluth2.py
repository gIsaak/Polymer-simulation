import numpy as np
import random
from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    '''
    Defines function for data fitting. (end-to-end distance)

    In: x: x-axis values
        a,b,c: fit paramters
    Out: function values
    '''
    return a*(x-1)**b + c

def func075(x, a, c):
    '''
    Defines function for data fitting. (end-to-end distance)

    In: x: x-axis values
        a,c: fit paramters
    Out: function values
    '''
    return a*(x-1)**0.75 + c

def func35(x, b, c):
    '''
    Defines function for data fitting. (radius of gyration)

    In: x: x-axis values
        b,c: fit paramters
    Out: function values
    '''
    return (x/6)**(b) + c

def fitFunc(tot_dist, A_w_err,A_w_RR,A_w_err_RR, start, bb, step, dstep, pop_size=None, plot=True):
    '''
    Method to plot the end-to-end distance data of Rosenbluth and PERM
    in a single plot.

    In: tot_dist (array): End-to-end distance values from algorithm 1
        A_w_err(array): Errors of values from algorithm 1 
        A_w_RR (array): End-to-end distance values from algorithm 2
        A_w_err_RR(array): Errors of values from algorithm 2 
        start,bb,step (integers): Paramerters defining the simulation bead numbers
        dstep (int): Every dteps'th data point will be plotted
        pop_size (array): Optional array to plot
        plot (boolean): Default: True: a plot will be generated
                        Falso: no plot generated
    Out: param_A1 (touple): Fitting parameters obtained from algorithm 1
         param_A2 (touple): Fitting parameters obtained from algorithm 2
    '''
    tot_dist = np.trim_zeros(tot_dist, 'b')

    xdata = np.asarray(range(start,bb,step))
    xdata = xdata[:len(tot_dist)]
    ydata = tot_dist
    
    param_A1, pcov = curve_fit(func, xdata, ydata)
    param_A2, pcov = curve_fit(func, xdata, A_w_RR)
    f_param, f_pcov = curve_fit(func075, xdata, A_w_RR)
    
    if plot:
        if pop_size is not None:
            plt.plot(xdata[::dstep],pop_size[::dstep],marker='o',markersize=2, linewidth=0,color='b',label='population size')
        plt.plot(xdata[::dstep], A_w_RR[::dstep]**2, marker='x',markersize=7, color='g', linewidth=0,label='PERM')
        plt.errorbar(xdata[::dstep], A_w_RR[::dstep]**2, yerr=A_w_err_RR[::dstep]**2, fmt='none', color='g', capsize=0)
        plt.plot(xdata[::dstep], ydata[::dstep]**2, marker='.',markersize=5, color='m', linewidth=0,label='Rosenbluth')
        plt.errorbar(xdata[::dstep], ydata[::dstep]**2, yerr=A_w_err[::dstep]**2, fmt='none', color='m', capsize=3)
        plt.plot(xdata, func075(xdata, *f_param)**2, 'r--',linewidth=2,label='$a(N-1)^{0.75}+c$')
        
        plt.xlabel('$N$')
        plt.ylabel('$R_{N}^{2}$')
        plt.title('End-to-end distance squared')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
#        plt.savefig('PERM_looooool.png',dpi=500)
        plt.show()       
    return param_A1,param_A2

def fitFuncRg(tot_dist, r_gyration_err, r_gyration_RR,r_gyration_err_RR, start, bb, step, dstep, plot=True):
    '''
    Method to plot the radius of gyration data of Rosenbluth and PERM
    in a single plot.

    In: tot_dist (array): Radius of gyration values from algorithm 1
        r_gyration_err(array): Errors of values from algorithm 1 
        r_gyration_RR (array): Radius of gyration values from algorithm 2
        r_gyration_err_RR(array): Errors of values from algorithm 2 
        start,bb,step (integers): Paramerters defining the simulation bead numbers
        dstep (int): Every dteps'th data point will be plotted
        plot (boolean): Default: True: a plot will be generated
                        Falso: no plot generated
    Out: param_A1 (touple): Fitting parameters obtained from algorithm 1
         param_A2 (touple): Fitting parameters obtained from algorithm 2
    '''
    tot_dist = np.trim_zeros(tot_dist, 'b')

    xdata = np.asarray(range(start,bb,step))
    xdata = xdata[:len(tot_dist)]
    ydata = tot_dist
    
    param_A1, f_pcov = curve_fit(func35, xdata, ydata)
    param_A2, f_pcov = curve_fit(func35, xdata, r_gyration_RR)

    if plot:
        plt.plot(xdata[::dstep], ydata[::dstep]**2, marker='.',markersize=5, color='m', linewidth=0,label='Rosenbluth')
        plt.errorbar(xdata[::dstep], ydata[::dstep]**2, yerr=r_gyration_err[::dstep]**2,fmt='none', color='m',capsize=5)
        
        plt.plot(xdata[::dstep], r_gyration_RR[::dstep]**2, marker='x',markersize=7, color='g', linewidth=0,label='PERM')
        plt.errorbar(xdata[::dstep], r_gyration_RR[::dstep]**2, yerr=r_gyration_err_RR[::dstep]**2, fmt='none', color='g',capsize=5)
        
        plt.plot(xdata, func35(xdata, *param_A2)**2,'r--',linewidth=2,label='$(^N/_6)^{b}+c$')

        plt.xlabel('$N$')
        plt.ylabel('$R_{g}^{2}$')
        plt.title('Radius of gyration squared')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
#        plt.savefig('PERM_r_gyration_3.png',dpi=500)
        plt.show()
    return param_A1, param_A2

def getTotDist(polymer):
    '''
    Gets end-to-end distance of a polymer

    In: polymer (nparray L x 2): polymer of (arbitraty) length L
    Out: dist (float): distance of the last to the first bead (sits at origin)
    '''    
    dist = 0
    dist = np.sqrt((polymer[-1,0])**2 + (polymer[-1,1])**2)
    return dist

def getTrialPos(polymer,res):
    '''
    Gets trial positions for new bead

    In: polymer (nparray L x 2): polymer of (arbitraty) length L
        res (integer): number of trial position along the arc
    Out: pos (nparray res x 2): array of trial positions
    '''
    # get position of last bead
    x_last = polymer[-1,0]
    y_last = polymer[-1,1]
    # compute angles (units of 2pi)
    init_angle = random.uniform(0, 1)
    divs = np.linspace(1.0/res, 1, res)
    angles = (init_angle + divs)%1
    # get trial positions
    pos = np.zeros(shape=(res,2))
    pos[:,0] = x_last + np.cos(angles*2*pi)
    pos[:,1] = y_last + np.sin(angles*2*pi)
    return pos

def getEj(polymer, pos, sigma, eps):
    '''
    Calculates E for new bead in every trial position

    In: polymer (nparray L x 2): polymer of (arbitrary) length L
        pos (nparray res x 2): trial positions form getTrialPos
        sigma (float): defines minimum of interaction potential
        eps (float): defines energy scale of interaction potential
    Out: Ej (nparray res): energy of every pos
    '''
    rotated = np.rot90(pos[:,:,None],1,(0,2))
    dist_comp = polymer[:,:,None] - rotated #broadcasting
    dist = np.sqrt(np.sum(dist_comp**2, axis=1)) #distances matrix
    inv_dist6 = np.reciprocal((dist/sigma)**6)
    Ej_mat = 4*eps*(inv_dist6**2 - inv_dist6)
    Ej = np.sum(Ej_mat, axis=0)
    return Ej

def getWeight(polymer, parameters):
    '''
    Calculates weights for all trial positions

    In: polymer (nparray L x 2): polymer of (arbitrary) length L
        T (float): Temperature
    Out: pos (nparray res x 2): trial positions form getTrialPos
         wj (nparray res): array of trial positions boltzmann weights
         W (float): sum of wj
    '''
    T, eps, sigma, res = parameters
    pos =  getTrialPos(polymer, res)
    Ej = getEj(polymer, pos, sigma, eps)
    wj = np.exp(-Ej/T)
    W = np.sum(wj)
    return pos, wj, W

def playRoulette(pos, wj, W):
    '''
    Roulette-wheel algorithm to select new position

    In: pos (nparray res x 2): trial positions form getTrialPos
        wj (nparray res): weights of trial positions
        W (float): sum of weights
    Out: (nparray 1 x 2): coordinates of chosen position
    '''
    j, step = 0, wj[0] #counter
    shot = random.uniform(0, W) #random number extraction
    while step < shot:
        j += 1
        step += wj[j]
    return pos[j,:]

def genPoly(polymer, N, parameters):
    '''
    Generates polymer of size N at temperature T

    In: polymer (nparray 2 x 2): input two bead polymer
        N (int): final size of polymer
    Out: polymer (final polymer of size N)
         pol_weight (float): final polymer weight
    '''
    pol_weight = 1
    for i in range(N-2):
        pos, wj, W = getWeight(polymer,parameters)
        new_bead = playRoulette(pos, wj, W)
        polymer = np.vstack((polymer, new_bead)) #bead added
        pol_weight *= W
    return polymer, pol_weight

# Hic sunt leones (PERM)

# TODO need to find a smarted implementation for this function
def getAvWeight(population):
    '''
    Returns average weight of polymer population
    In: population (list)
    Out: (float): avergae weight of population
    '''
    av_weight = 0
    for poly in population:
        av_weight += poly[1]
    return av_weight/len(population)

def initializePopulation(initial_pop_size,parameters):
    '''
    Initializes a population of initial_pop_size number of 3-bead polymers
    In: initial_pop_size (int): size of population after initialization
    Out: population (list): grown population of 3-beads polymers
                            Each element in the list is a list [polymer, pol_weight]
         av_weight3 (float): average polymer weight of population of 3-bead polymers
    '''
    polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
    population = []
    for i in range(initial_pop_size):
        population.append(list(genPoly(polymer, 3, parameters)))
    return population

def addBead(poly,parameters):
    '''
    Adds one bead to a polymer and updates its weight

    In: poly (list): Element of population [polymer, pol_weight]
                     polymer (nparray L+1 x 2) (final polymer of size N)
                     pol_weight (float): final polymer weight
        pol_weight (float): final polymer weight
    Out: polymer (nparray L+1 x 2) (final polymer of size N)
         pol_weight (float): final polymer weight
    '''
    polymer, pol_weight =  poly[0], poly[1]
    pos, wj, W = getWeight(polymer,parameters)
    new_bead = playRoulette(pos, wj, W)
    polymer = np.vstack((polymer, new_bead)) #bead added
    pol_weight *= W
    return polymer, pol_weight

# TODO find way to speedup ugly nested for loos
def grow(population, step_size, up_th, low_th, parameters):
    '''
    Adds step_size number of beads to each polymer in the population,
    calculates up_lim and low_lim and decides whether to prune or split
    In:  population (list): population polymers
                            Each element in the list is a list [polymer, pol_weight]
        step_size (int): number of beads to add before pruning or splitting
        up_th (float): upper threshold
        low_th (float): lower threshold
    '''
    # add step_size beads
    for i in range(step_size):
        for i, poly in enumerate(population):
            population[i] = list(addBead(poly, parameters))
    # gets up_lim and low_lim
    av_weight = getAvWeight(population)
    for i, poly in enumerate(population):
        if poly[1] > up_th*av_weight:
            #split
            population[i][1] /= 2.0
            poly[1] /= 2.0
            population.append(poly)
        elif poly[1] < low_th*av_weight:
            # prune
            population[i][1] *= 2.0
            if random.uniform(0, 1) < 0.5:
                del population[i]
    return population

def getS(population):
    '''
    Extracts data from population list to compute the weighted average of 
    the end-to-end distance S and its error.
    In:  population (list): grown population of polymers
                            Each element in the list is a list [polymer, pol_weight]
    Out: S (float): weighted average of end-to-end distance
         S_err (float): error on S
    '''
    end_dist_list = list()
    weight_list = list()
    for poly in population:
        S2_poly = np.sum(poly[0][-1,:]**2)
        end_dist_list.append(S2_poly)
        weight_list.append(poly[1])

    end_dist_arr = np.sqrt(np.asarray(end_dist_list))
    weights = np.asarray(weight_list)
    weights = weights/np.sum(weights)
    
    S, S_err = weightedAverageErr(end_dist_arr,weights)

    return S, S_err

def getRg(population):
    '''
    Extracts data from population list to compute the weighted average of 
    the radius of gyration Rg and its error.
    In:  population (list): grown population of polymers
                            Each element in the list is a list [polymer, pol_weight]
    Out: Rg (float): average square of the radius of gyration Rg
         Rg_err (float): error on Rg
    '''
    r_gyration_list = list()
    weight_list = list()
    for poly in population:
        R_poly = np.sqrt(np.sum(poly[0]**2, axis=1))
        Rcm = np.average(R_poly)
        Rg2_poly = np.average((R_poly - Rcm)**2)
        weight_list.append(poly[1])
        r_gyration_list.append(Rg2_poly)
    
    Rg_arr = np.sqrt(np.asarray(r_gyration_list))
    weights = np.asarray(weight_list)
    weights = weights/np.sum(weights)
    
    Rg, Rg_err = weightedAverageErr(Rg_arr,weights)

    return  Rg, Rg_err

def weightedAverageErr(arr, weights):
    '''
    Calculates a weighted average and the corresponding error for input data.
    In:  arr (numpy array): array holding the observable
         weights (numpy array): array holding the corresponding weights
    
    Out: omean (float): weighted average of the observable
         err (float): error on the observable's weighted mean
    '''
    o_mean = np.sum(arr*weights)/np.sum(weights)
    n = len(arr)
    err = np.sqrt((n/(n-1))*np.sum((weights**2)*(arr-o_mean)**2))
    return o_mean, err
    


