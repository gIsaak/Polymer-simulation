import numpy as np
import random
from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from matplotlib.ticker import NullFormatter  

def func(x, a, b, c):
    return a*x**b + c

def fit_func_std(tot_dist, A_std, plot=True):
    tot_dist = np.trim_zeros(tot_dist, 'b')
    
    xdata = np.arange(len(tot_dist))
    ydata = tot_dist
    param, pcov = curve_fit(func, xdata, ydata)
    
    if plot:
#        plt.plot(xdata, ydata, 'b-', label='bead distance')
        plt.errorbar(xdata, ydata**2, yerr=A_std, label='bead distance')
#        plt.plot(xdata, func(xdata, *param), 'r-', label='fit: a={:5.3f}, b={:5.3f}, c={:5.3f}'.format(param[0],param[1],param[2]))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Beads')
        plt.ylabel('$R^2$')
        plt.legend()

        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.show()
    return param

def fit_func(tot_dist, plot=True):
    tot_dist = np.trim_zeros(tot_dist, 'b')
    
    xdata = np.arange(len(tot_dist))
    ydata = tot_dist
    param, pcov = curve_fit(func, xdata, ydata)
    
    if plot:
        plt.plot(xdata, ydata, 'b-', label='bead distance')
        plt.plot(xdata, func(xdata, *param), 'r-', label='fit: a={:5.3f}, b={:5.3f}, c={:5.3f}'.format(param[0],param[1],param[2]))
#        plt.yscale('log')
#        plt.xscale('log')
        plt.legend()
        plt.show()
    return param

def get_tot_dist(polymer):
    # since initial particle is at (0,0)its practically the length of position vector of last bead
    dist = 0
    dist = abs(np.sqrt((polymer[-1,0]-polymer[0,0])**2 + (polymer[-1,1]-polymer[0,1])**2))# + 1
    return dist

def getTrialPos(polymer,res):
    '''
    Gets trial positions for new bead
    Needs res (global variable)

    In: polymer
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

def getEj(polymer, pos, res, eps, sigma):
    '''
    Calculates E for new bead in every trial position
    Needs global variables eps, sigma, res

    In: polymer
        pos (nparray res x 2): trial positions form getTrialPos
    Out: Ej (nparray res): energy of every pos
    '''
    rotated = np.rot90(pos[:,:,None],1,(0,2))
    distComp = polymer[:,:,None] - rotated #broadcasting
    dist = np.sqrt(np.sum(distComp**2, axis=1)) #distances matrix
    invDist6 = np.reciprocal((dist/sigma)**6)
    Ej_mat = 4*eps*(invDist6**2 - invDist6)
    Ej = np.sum(Ej_mat, axis=0)
    return Ej

def get_weight(polymer, T, res, eps, sigma):
    '''
    Calculates weights for all trial positions
    Needs res (global variable)

    In: Ej (nparray res): from getEj
        T (float): temperature
    Out: pos (nparray res x 2): trial positions form getTrialPos
         wj (nparray res): array of trial positions boltzmann weights
         W (float): sum of wj
    '''
    pos =  getTrialPos(polymer, res)
    Ej = getEj(polymer, pos, res, eps, sigma)
    wj = np.exp(-Ej/T)
    W = np.sum(wj)
    return pos, wj, W

def play_roulette(pos, wj, W):
    '''
    Roulette-wheel algorithm to select new position

    In: pos (nparray res x 2): trial positions form getTrialPos
        wj (nparray res): weights of trial positions
        W (float): sum of weights
    Out: j index of chosen trial position
    '''
    j, step = 0, wj[0] #counter
    shot = random.uniform(0, W) #random number extraction
    while step < shot:
        j += 1
        step += wj[j]
    return pos[j,:]

def genPoly(polymer, r_dict):
    '''
    Generates polymer of size N at temperature T
    Needs eps, sigma, res global varialbes

    In: polymer (input two bead polymer)
        N (int): polymer size minus 2 due to starting beads
        T (float): temperature
    Out: polymer (final polymer of size N)
         polWeight (float): final polymer weight
    '''
    
    #######################
    ### Load Dictionary ###
    #######################
    N           = r_dict['number_of_beads']
    T           = r_dict['temperature']
    res         = r_dict['resolution']
    sigma       = r_dict['sigma']
    eps         = r_dict['epsilon']
    tot_dist = np.zeros(shape=(N-2))
    tot_polWeight = np.zeros(shape=(N-2))
    
    polWeight = 1
    for i in range(N-2):
        pos, wj, W = get_weight(polymer, T, res, eps, sigma)
        newBead = play_roulette(pos, wj, W)
        polymer = np.vstack((polymer, newBead)) #bead added
        polWeight *= W
        
        tot_polWeight[i] = polWeight
        tot_dist[i] = get_tot_dist(polymer)
    
        
    return polymer, polWeight, tot_dist, tot_polWeight



