'''
Chapter 2: From words to models

    @Zeming 

'''

import numpy as np 
import matplotlib.pyplot as plt 

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255

def random_walk_model(  drift =0., sdrw =.3, criterion=3,
                        n_reps = 10000, n_samples = 2000):
    '''
    
    '''
    # data storages
    latencies = np.zeros([n_reps,])
    responses = np.zeros([n_reps,])
    evidence  = np.zeros([ n_reps, n_samples+1])
    colors    = [ Blue, Red, Green, Yellow, Purple]

    ## Simulate the data 
    for i in range( n_reps):
        # generate a sequence of shifts at different timesteps.
        # note that the shifts at the very first timestep is 0
        shifts = np.insert( np.random.normal( drift, sdrw, n_samples), 0, 0)
        # the ending position at the prevsiout timestep t-1 is 
        # the initial of the current shift t, 
        evidence[ i] = np.cumsum( shifts)
        # find the first sample that run out of the bound (criterion)
        p = np.where( abs(evidence[i,]) > criterion)[0][0]
        # record the data 
        responses[ i] = np.sign( evidence[ i, p])
        latencies[ i] = int(p) 

    ## Plot up to 5 random-walk
    # creat timestep axis 
    tbpn = np.min( [n_reps, 5])
    T = int(np.max(latencies[:tbpn]) + 20)
    timesteps = np.arange( T)
    for i in range(tbpn):
        data = evidence[ i, :int(latencies[i])]
        x = np.arange( 0, len( data))
        plt.plot( x, data, color=colors[i]) 
    plt.plot( timesteps, [criterion] * T, 'k--')
    plt.plot( timesteps, [-criterion] * T, 'k--')
    plt.xlabel( 'Time')
    plt.ylabel( 'Evidence')
    plt.xlim( [ 0, T]) 
    plt.ylim( [ -criterion-.5, criterion+.5])
    plt.show()

    
if __name__ == '__main__':

    random_walk_model()



    
