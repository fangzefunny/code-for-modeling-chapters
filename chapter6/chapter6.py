import os 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import beta

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

def beta_curve():
    plt.figure( figsize=( 4, 4))
    x = np.linspace( 0, 1, 40)
    plt.plot( x, beta.pdf( x, 2, 4), color='k', label='Johnnie')
    plt.plot( x, beta.pdf( x, 8, 16), '--', color='k', label='Jane')
    plt.legend( ['Johnnie', 'Jane'])
    plt.xlabel( 'x')
    plt.ylabel( 'Probability Density')
    plt.savefig( f'{path}/Fig1-beta distributions')


if __name__ == '__main__':

    beta_curve()