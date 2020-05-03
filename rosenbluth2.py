import numpy as np
import random
from math import pi, sin, cos
import matplotlib.pyplot as plt

def getTrialPos(polymer):
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

def getEj(polymer, pos):
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

def get_weight(polymer, T):
    '''
    Calculates weights for all trial positions
    Needs res (global variable)

    In: Ej (nparray res): from getEj
        T (float): temperature
    Out: pos (nparray res x 2): trial positions form getTrialPos
         wj (nparray res): array of trial positions boltzmann weights
         W (float): sum of wj
    '''
    pos =  getTrialPos(polymer)
    Ej = getEj(polymer, pos)
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

def genPoly(polymer, N, T):
    '''
    Generates polymer of size N at temperature T
    Needs eps, sigma, res global varialbes

    In: polymer (input two bead polymer)
        N (int): polymer size
        T (float): temperature
    Out: polymer (final polymer of size N)
         polWeight (float): final polymer weight
    '''
    polWeight = 1
    for i in range(N):
        pos, wj, W = get_weight(polymer, T)
        newBead = play_roulette(pos, wj, W)
        polymer = np.vstack((polymer, newBead)) #bead added
        polWeight *= W
    return polymer, polWeight

# MAIN PROGRAM
# global variables sigma, eps, res
sigma = 1.0
eps = 1.0
res = 100

N = 250
T = 1.0

polymer = np.array(([0.0, 0.0], [1.0, 0.0]))
polymer, polWeight = genPoly(polymer, N, T)

fig = plt.figure()
plot = plt.plot(polymer[:,0],polymer[:,1], '-bo')

plt.show()
