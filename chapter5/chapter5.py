'''
Chapter 5: Combine information from multiple participants

    @Zeming 

'''
import os
from re import M
import numpy as np
from numpy.lib.function_base import disp
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, norm
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

def weib_qdev( x, q_emp, q_p):
    if np.min(x) <= 0:
        return 1e7
    else:
        q_pred = weibull_min.ppf( q_p, x[2], loc=x[0], scale=x[1])
        return np.sqrt( np.mean( (q_pred - q_emp)**2))

def weib_inddev( x, rts):
    if ( np.min(x) <= 0) or (np.min( rts < x[0])):
        return 1e7
    else:
        like = weibull_min.pdf( rts-x[0], x[2], loc=0, scale=x[1])
        return np.sum( -2 * np.log( like + 1e-18))

def fit( n_subj=30, n_obs=20, q_p = np.arange( .1, 1.1, .2), 
         mode='vincent avg', seed=1234):
    '''Fit function

    if mode == 'vincent avg', do vincent averaging 
    elif mode == 'ind', fit to individual data 
    '''
    ## fix the seed 
    np.random.RandomState( seed)

    ## Simulate params 
    shift = np.random.normal( 250, 50, n_subj)
    scale = np.random.normal( 200, 50, n_subj)
    shape = np.random.normal( 2, .25, n_subj)
    params = np.vstack( [ shift, scale, shape])

    ## Simple average
    print( np.mean( params, axis=1))

    ## Do vincent average 
    rweibull = lambda x: weibull_min.rvs( x[2], 
                loc=0, scale=x[1], size=n_obs) + x[0]
    quantile = lambda x: np.quantile( x, q=q_p)
    param_lst = [ params[ :, i] for i in range( n_subj)]
    dat = list( map( rweibull, param_lst))
    
    if mode == 'vincent avg':
        ## Fit via shift weibull to averaged quantiles 
        kk  = np.vstack( list( map( quantile, dat)))
        vinq = np.mean( kk, axis=0)
        res  = minimize( weib_qdev, [ 225, 225, 1],
                        args=( vinq, q_p))
        print(res.x)

    elif mode == 'ind':
        ## Fit via shift weibull to individual 
        fit_ind = lambda x: minimize( weib_inddev, [ 100, 225, 1],
                        args=( x), options={ 'disp': False})
        res = list( map( fit_ind, dat))
        parest = np.vstack( list( map( lambda f: f.x, res)))

        print( f'''
                mean: { np.mean( parest, 0)}
                std : { np.std( parest, 0)}
                ''')

def fit_hier( N=1000, pShort=.3, gen_pars=[ [100,10], [150, 20]],
              mode = 'mix', seed=1540614451):
    '''Indentify different group 

        Sacadic eye movements in eye-tracking experiemnt
        two types of sacadic depends on saccadic latencies: 
            --normal: longer latencies 
            --express: short latencies 

        Inputs:
            N: number of data point 
            pShort: a probability of an express sacadic
            gen_pars: groud truth for simulation 
    '''
    ## Assign a random generator 
    rng = np.random.RandomState( seed)
    
    ## Simulate data 
    whichD = rng.choice( [ 0, 1], size=N, p=[ pShort, 1 - pShort])
    dat = np.hstack( list( map( lambda x: norm.rvs( loc=gen_pars[x][0], scale=gen_pars[x][1]),
                     whichD)))

    ## function needed in EM 
    def weighted_mean( x, w, mu=None):
        mu = mu if mu else np.mean(x)
        return np.sum( w * x) / np.sum( w)

    def weighted_sd( x, w, mu=None):
        mu = mu if mu else np.mean(x)
        wvar = np.sum( w * ( x - mu) ** 2) / np.sum( w)
        return np.sqrt( wvar) 

    ## Init the iteration
    # guess parameters
    mu1 = np.mean( dat) * .8
    mu2 = np.mean( dat) * 1.2
    sd1 = sd2 = np.std( dat)
    ppi = .5  # probability that the data is belongs to the 2nd distribution 
    oldppi = 0  

    ## Run EM until converge
    while ( abs( ppi - oldppi) > 1e-5):
        # cache old data 
        oldppi = ppi 
        # E step
        resp = ppi * norm( mu2, sd2).pdf( dat) / \
               ( (1 - ppi) * norm( mu1, sd1).pdf( dat) 
                + ppi * norm( mu2, sd2).pdf( dat)) 
        # M step 
        mu1 = weighted_mean( dat, 1-resp)
        mu2 = weighted_mean( dat, resp)
        sd1 = weighted_sd( dat, 1-resp, mu1)
        sd2 = weighted_sd( dat, resp, mu2)
        ppi = np.mean( resp)
    
    ## Visualize
    print( f'True params: {gen_pars}')
    print( f'Fitted params: [[ {mu1:.2f}, {sd1:.2f}], [ {mu2:.2f}, {sd2:.2f}]]')
    dat = np.sort(dat)
    x = np.arange( 50, 250 , .1)
    n_gap = 5
    bins = np.arange( 50, 250, n_gap)
    plt.hist( np.hstack( [ 
        norm( mu1, sd1).rvs(int((1 - ppi) * N)),
        norm( mu2, sd2).rvs(int(ppi * N))]), 
        bins=bins,
        density=True,
        facecolor='gray', edgecolor='k')
    plt.plot( x, (1 - ppi) * norm.pdf( x, loc=mu1, scale=sd1),
                color=Red)
    plt.plot( x, ppi * norm.pdf( x, loc=mu2, scale=sd2),
                color=Blue)
    plt.xlabel( 'RTs')
    plt.ylabel( 'Density')
    plt.savefig( f'{path}/fig2-Fitted distribution.png', dpi=dpi)


if __name__ == '__main__':

    ## fit data with flat structure 
    fit( mode='vincent avg')
    fit( mode='ind')

    ## fit data with hierarchical structure 
    fit_hier( mode='mix')



