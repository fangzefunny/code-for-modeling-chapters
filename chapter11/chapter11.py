'''
Chapter 11: Bayesian Modeling Comparison Using Bayes Factors

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
from scipy.stats import binom
# integration functions 
from scipy.integrate import nquad


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

#================================
#     Numerical Intergration
#================================

def numerical_int( ):

    ## Define values for example
    tlags = np.array([ 0, 1, 5, 10, 20, 50])
    nitems = 40 
    a, b, alpha = .1, .95, .2 

    ## Simulate data 
    decay_fn = lambda t: a+(1-a)*b*np.exp(-alpha*t)
    ps = [ decay_fn(t) for t in tlags]
    nrecalls = binom.rvs( n=nitems, p=ps)
    
    ## Functions
    def exp_like( a, b, alpha, tlags, y, n):
        ps = a+(1-a)*b*np.exp(-alpha*tlags)
        return np.prod( binom.pmf(y, n=n, p=ps))
    
    def pow_like( a, b, beta, tlags, y, n):
        ps = a+(1-a)*b*((tlags+1)**(-beta))
        return np.prod( binom.pmf(y, n=n, p=ps))

    ## Calculate Bayes Factors 
    exp_evd = nquad( exp_like, [(0, .2), (0, 1), (0, 1)], 
                args=( tlags, nrecalls, nitems))[0]
    pow_evd = nquad( pow_like, [(0, .2), (0, 1), (0, 1)], 
                args=( tlags, nrecalls, nitems))[0]
    BF = exp_evd / pow_evd
    
    return BF




if __name__ == '__main__':

    ## numerical intergration 
    print( numerical_int())