'''
Chapter 4: Maximum Likelihood Pamameters Estimation

    @Zeming 

'''
import os
from re import M
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import binom, norm 
from scipy.optimize import minimize


# find the current path
path = os.path.dirname(os.path.abspath(__file__))

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors    = [ Blue, Red, Green, Yellow, Purple]

# image dpi
dpi = 150

def rswald( t, a, m ,T):
    return a / np.sqrt( 2 * np.pi * ( t - T) **3
            ) * np.exp( - ( a - m * ( t - T))**2 
            ) / ( 2 * ( t- T)) 


def GCMpred( probe, exemplars, c, w):
    '''GCM pred
    '''
    l2_dist = lambda x: np.sqrt( np.sum( w * (x - probe)**2))
    exp_law = lambda x: np.sum( np.exp( -c * x))
    dist = [ np.array(list(map( l2_dist, list(ex)))) for ex in exemplars]
    sumsim = np.array( list(map( exp_law, dist)))
    return sumsim / np.sum( sumsim) # normalize 

    
def GCM_and_binom( n_sub, pa=.968, c=4, w=[ .19, .12, .25, .45]):
    '''Linking GCM and binomial function

        N: All observations 
        N_A:
    '''
    ## Define value 
    N = n_sub * 2     # tot actions
    N_A = int(N * .968)    # num of act A 

    ## Load stimulus 
    stim = pd.read_csv( f'{path}/faceStim.csv', index_col=False).values
    exemplars = [ stim[ :5, :], stim[ 5:10, :]]
    preds = GCMpred( stim[ 0, :], exemplars, c, w)
    like  = binom.pmf( N_A, N, preds[0])

def GCMprednoisy( probe, exemplars, c, w, sigma, b):
    '''
    p ~ N(p; p(a)-p(b)-b, sigma)
    '''
    l2_dist = lambda x: np.sqrt( np.sum( w * (x - probe)**2))
    exp_law = lambda x: np.sum( np.exp( -c * x))
    dist = [ np.array( list(map( l2_dist, list(ex)))) for ex in exemplars]
    sumsim = np.array( list(map( exp_law, dist)))
    r_prob = norm.cdf( sumsim[0]-sumsim[1]-b, loc=0, scale=sigma)
    return np.array( [ r_prob, 1 - r_prob])

def GCMmutil( theta, stim, exemplars, data, N , retpreds):
    '''A function to get deviance from GCM

    '''
    ## Get storages 
    nDat = len(data)
    dev  = np.ones( [ nDat,]) + np.nan 
    preds = np.ones_like( dev) + np.nan

    ## assign parameter 
    c = theta[ 0]
    w = np.ones([4,]) + np.nan 
    w[0] = theta[1]
    w[1] = theta[2] * ( 1 - w[0])
    w[2] = theta[3] * ( 1 - np.sum(w[:2]))
    w[3] = ( 1 - np.sum(w[:3]))
    sigma = theta[4]
    b = theta[5]

    ## forward 
    for i in range(nDat):
        p = GCMprednoisy( stim[i, :], exemplars, c, w, sigma, b)
        dev[i] = - 2 * binom.logpmf( data[i], N, p[0])
        preds[i] = p[0]

    ## Output 
    if retpreds:
        return preds 
    else: 
        return np.sum( dev)
    
def fit_params_MLE():

    N = 2 * 40
    stim = pd.read_csv( f'{path}/faceStim.csv', header=None).values
    exemplars = [ stim[ :5, :], stim[ 5:10, :]]
    with open( f'{path}/facesDataLearners.txt' )as handle:
        data = handle.readlines()
    data = [ int(np.ceil(float(datum[:-2])*N)) for datum in data]
    bestfit = 10000

    # init w list 
    ws = [ [ .25, .5, .75]] * 3
    bnds = [ ( 0, 10), ( 0, 1), ( 0, 1), ( 0, 1), ( 0, 10), ( -5, 5)]
    for w1 in ws[0]:
        for w2 in ws[1]:
            for w3 in ws[2]:
                print( f'Start with w1={w1}, w2={w2}, w3={w3}')
                theta0 = [ 1, w1, w2, w3, 1, .2]
                fit_res = minimize( GCMmutil, theta0, 
                                    args=( stim, exemplars, data, N, False),
                                    bounds=bnds,
                                    options={'disp': False})
                print( f'Fitted params: {fit_res}') 
                if fit_res.fun <= bestfit:
                    bestfit = fit_res.fun  
                    bestres = fit_res
    ## Print the results
    #param = [ 2.55, .37, .005, .61, .01, .079]
    preds = GCMmutil( bestres.x, stim, exemplars, data, N, True)
    ## Plot data and best-fitting predictions
    _ = plt.figure( figsize=( 4, 4))
    plt.scatter( np.array(data)/N, preds,
                s=60, facecolors='none', edgecolors=Red)
    plt.xlabel( 'Data')
    plt.ylabel( 'Predictions')
    plt.savefig( f'{path}/Fig10-GCM fit diagnosis', dpi=dpi)
    theta = bestres.x 
    w = np.ones([4,]) + np.nan
    w[0] = theta[1]
    w[1] = theta[2] * ( 1 - w[0])
    w[2] = theta[3] * ( 1 - np.sum(w[:2]))
    w[3] = ( 1 - np.sum(w[:3]))
    print( w)

if __name__ == '__main__':

    fit_params_MLE()