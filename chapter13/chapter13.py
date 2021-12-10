'''
Chapter 13: Neural Network Models

    @Zeming 

'''

## Import packages
# basic pkgs
import os
import numpy as np 
# visualization pkgs 
import matplotlib.pyplot as plt
import seaborn as sns
# stats distribution
from scipy.stats import norm 


# find the current path
path = os.path.dirname(os.path.abspath(__file__))

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors    = [ Blue, Red, Green, Yellow, Purple]
sns.set_style("whitegrid", {'axes.grid' : False})

# image dpi
dpi = 250

## Norm and Cosine similarity
vnorm  = lambda x: np.sqrt(np.sum(x**2))
cosine = lambda a, b: np.dot( a, b) / (vnorm(a) * vnorm(b))

#==========================
#     Hebbian model
#==========================

def hebb_model( ):

    ## Stimuli representation
    stim = np.array([[ 1, -1,  1, -1],
                     [ 1,  1,  1,  1]])
    resp = np.array([[ 1,  1, -1, -1],
                     [ 1, -1, -1,  1]])

    ## Init weight 
    p, n, m = 2, 4, 4
    alpha = .25 
    W = np.zeros( [ m, n])

    ## Learning 
    for pair in range(p):
        for i in range(n):
            for j in range(m):
                W[ i, j] += alpha * stim[ pair, j]\
                                  * resp[ pair, i]
    
    ## Test 
    o = np.zeros( [m,])
    for i in range(m):
        for j in range(n):
            o[i] += W[ i, j] * stim[ 0, j]

    return cosine( o, resp[ 0, :])

#====================================
#     Hebbian model matrix form 
#===================================

def hebb_mat():

    ## Define values
    n, m = 100, 50  # n: input, m: output 
    listLen, nRep = 20, 100 
    alpha = .25 
    stimSimSet = np.array([ 0, .25, .5, .75, 1])
    accuracy = np.zeros_like( stimSimSet)

    ## Start simulation
    for _ in range( nRep):

        # train stim for learning task 
        W = np.zeros( [ m, n])
        stim1 = np.sign( np.vstack([ norm.rvs(size=n) 
                        for _ in range(listLen)]))
        resp1 = np.sign( np.vstack([ norm.rvs(size=m) 
                        for _ in range(listLen)]))

        # Learning 
        for i in range( listLen):
            o, c = resp1[ [i], :], stim1[ [i], :]
            W += alpha * o.T @ c 
    
        # test stim at different levels of
        # similarity
        for stim_idx, stimSim in enumerate(stimSimSet):
            # create test stim 
            mask = np.vstack( [ np.random.rand(n) < stimSim 
                            for _ in range(listLen)]) 
            svec = np.sign( np.vstack([ norm.rvs(n) 
                            for _ in range(listLen)]))
            stim2 = mask*stim1 + (1-mask)*svec
            # Test the trained model
            acc = 0
            for j in range( listLen):
                c = stim2[ [j], :] #1n 
                o = c @ W.T  # 1n @ nm = 1m
                acc += cosine( o[0,:], resp1[j,:]) / listLen
            accuracy[ stim_idx] += acc / nRep 
        
    ## Visualization
    fig = plt.figure(figsize=( 4.5, 4))
    plt.plot( stimSimSet, accuracy, 'o-', color=Blue)
    plt.xlabel( 'Stim-Cue Similarity', fontsize=13)
    plt.ylabel( 'Cosine', fontsize=13)
    plt.xlim( [ -0.05, 1.05])
    plt.ylim( [ -0.05, .95])
    plt.tight_layout()
    plt.savefig( f'{path}/Fig4-Hebb_mat', dpi=dpi)

#=========================
#     Auto Associator   
#=========================


if __name__ == '__main__':

    ## numerical intergration 
    #print( hebb_model())

    ## hebb matrix example 
    hebb_mat()