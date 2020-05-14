import numpy as np
import random
from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a*(x)**b + c

def fitFunc(tot_dist, A_w_err, plot=True):
    tot_dist = np.trim_zeros(tot_dist, 'b')

    xdata = np.arange(len(tot_dist))*5
    ydata = tot_dist
    param, pcov = curve_fit(func, xdata, ydata)

    if plot:
#        plt.plot(xdata, ydata, 'b-', label='bead distance')
        plt.errorbar(xdata, ydata**2, yerr=A_w_err, label='bead distance err')
#        plt.plot(xdata, func(xdata, *param), 'r-', label='fit: a={:5.3f}, b={:5.3f}, c={:5.3f}'.format(param[0],param[1],param[2]))
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()
    return param

def getTotDist(polymer):
    # since initial particle is at (0,0)its practically the length of position vector of last bead
    dist = 0
    dist = np.sqrt((polymer[-1,0])**2 + (polymer[-1,1])**2)
    return dist

def getTrialPos(polymer):
    '''
    Gets trial positions for new bead
    Needs global variable res

    In: polymer (nparray L x 2): polymer of (arbitraty) length L
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

def getEj(polymer, pos):
    '''
    Calculates E for new bead in every trial position
    Needs global variables eps, sigma, res

    In: polymer (nparray L x 2): polymer of (arbitraty) length L
        pos (nparray res x 2): trial positions form getTrialPos
    Out: Ej (nparray res): energy of every pos
    '''
    rotated = np.rot90(pos[:,:,None],1,(0,2))
    dist_comp = polymer[:,:,None] - rotated #broadcasting
    dist = np.sqrt(np.sum(dist_comp**2, axis=1)) #distances matrix
    inv_dist6 = np.reciprocal((dist/sigma)**6)
    Ej_mat = 4*eps*(inv_dist6**2 - inv_dist6)
    Ej = np.sum(Ej_mat, axis=0)
    return Ej

def getWeight(polymer):
    '''
    Calculates weights for all trial positions
    Needs global variables T, eps, sigma, res

    In: polymer (nparray L x 2): polymer of (arbitraty) length L
    Out: pos (nparray res x 2): trial positions form getTrialPos
         wj (nparray res): array of trial positions boltzmann weights
         W (float): sum of wj
    '''
    pos =  getTrialPos(polymer)
    Ej = getEj(polymer, pos)
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

def genPoly(polymer, N):
    '''
    Generates polymer of size N at temperature T
    Needs T, eps, sigma, res, T global varialbes

    In: polymer (nparray 2 x 2): input two bead polymer
        N (int): final size of polymer
    Out: polymer (final polymer of size N)
         pol_weight (float): final polymer weight
    '''
    pol_weight = 1
    for i in range(N-2):
        pos, wj, W = getWeight(polymer)
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

def initializePopulation(initial_pop_size):
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
        population.append(list(genPoly(polymer, 3)))
    return population

def addBead(poly):
    '''
    Adds one bead to a polymer and updates its weight
    Needs eps, sigma, res, T global varialbes

    In: poly (list): Element of population [polymer, pol_weight]
                     polymer (nparray L+1 x 2) (final polymer of size N)
                     pol_weight (float): final polymer weight
        pol_weight (float): final polymer weight
    Out: polymer (nparray L+1 x 2) (final polymer of size N)
         pol_weight (float): final polymer weight
    '''
    polymer, pol_weight =  poly[0], poly[1]
    pos, wj, W = getWeight(polymer)
    new_bead = playRoulette(pos, wj, W)
    polymer = np.vstack((polymer, new_bead)) #bead added
    pol_weight *= W
    return polymer, pol_weight

# TODO find way to speedup ugly nested for loos
def grow(population, step_size, up_th, low_th):
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
            population[i] = list(addBead(poly))
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

def getS2(population):
    '''
    Computes the weighted average of the square of end-to-end distance S2 and its error
    from formed population
    In:  population (list): grown population of polymers
                            Each element in the list is a list [polymer, pol_weight]
    Out: S2 (float): average square of end-to-end distance
         S2_err (float): error on S2
    '''
    S2, S4, S2_err, tot_weight = 0, 0, 0, 0
    S2_poly, w_poly = 0, 0
    for poly in population:
        S2_poly = np.sum(poly[0][-1,:]**2)
        w_poly = poly[1]
        S2 += S2_poly * w_poly
        S4 += (S2_poly**2) * w_poly
        tot_weight += w_poly
    S2 = S2/tot_weight
    S4 = S4/tot_weight
    S2_err = np.sqrt(S4 - S2**2)
    return S2, S2_err

def getRg2(population):
    '''
    Computes the weighted average of the square of the radius of gyration Rg2
    and its error from formed population
    In:  population (list): grown population of polymers
                            Each element in the list is a list [polymer, pol_weight]
    Out: Rg2 (float): average square of the radius of gyration Rg2
         S2_err (float): error on Rg2
    '''
    Rg2, Rg4, Rg2_err, tot_weight = 0, 0, 0, 0
    Rg2_poly, w_poly = 0, 0
    for poly in population:
        R_poly = np.sqrt(np.sum(poly[0]**2, axis=1))
        Rcm = np.average(R_poly)
        Rg2_poly = np.average((R_poly - Rcm)**2)
        w_poly = poly[1]
        Rg2 += Rg2_poly*w_poly
        Rg4 += (Rg2_poly**2)*w_poly
        tot_weight += w_poly
    Rg2 = Rg2/tot_weight
    Rg4 = Rg4/tot_weight
    Rg2_err = np.sqrt(Rg4 - Rg2**2)
    return  Rg2, Rg2_err


# simulation parameters
eps, T, sigma, res, = 1, 1, 0.8, 6
# initial size of the population we start with
initial_pop_size = 1000

# need to think how to make this general, sorry im tired
step_size = 1 # bead added aech time before pruning/enrichment
num_steps = 22 # number of times we prune/enrich
# final number of beads
# N = 3 + step_size*num_steps

# This little simulation shows how the number of polymers changes with the number
# of beads when growing the population
pop_size = []
n = []
population = initializePopulation(initial_pop_size)
for i in range(num_steps):
    population = grow(population, step_size, 2.4, 0.3)
    pop_size.append(len(population))
    n.append(3 + i*step_size)

plt.plot(n, pop_size)
plt.xlabel('Number of beads')
plt.ylabel('Population size')
plt.show()

S2, S2_err = getS2(population)
print('S2 = ', S2, '+- ', S2_err)
Rg2, Rg2_err = getRg2(population)
print('Rg2 = ', Rg2, '+- ', Rg2_err)
