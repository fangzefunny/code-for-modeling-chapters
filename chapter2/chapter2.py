'''
Chapter 2: From words to models

    @Zeming 

'''
import os 
import numpy as np 
import matplotlib.pyplot as plt 

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

def basic_random_walk_model( drift =0., sdrw =.3, criterion=3,
                        n_reps = 10000, n_samples = 2000):
    '''A Basic Random Walk Model

    The relative position of the current step t and the
    last step t-1, is sample from a Gaussian distribution.
        shift ~ N( drift, sdrw)
    '''
    # data storages
    latencies = np.zeros([n_reps,])
    responses = np.zeros([n_reps,])
    evidence  = np.zeros([ n_reps, n_samples+1])
    
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
    
    return responses, latencies, evidence

def t2t_variability_model( drift =0., sdrw =.3, t2tsd=[ 0., .025],
                        criterion=3,
                        n_reps = 10000, n_samples = 2000):
    '''Variability model 
    '''
    # data storages
    latencies = np.zeros([n_reps,])
    responses = np.zeros([n_reps,])
    evidence  = np.zeros([ n_reps, n_samples+1])
    
    ## Simulate the data 
    for i in range( n_reps):
        # add a jitter to the start point  
        sp = np.random.normal( 0, t2tsd[0])
        # generate a jitter to the drift
        # mean of the Gaussian is not fixed
        dr = np.random.normal( drift, t2tsd[1])
        # generate a sequence of shifts at different timesteps.
        # note that the shifts at the very first timestep is 0
        shifts = np.insert( np.random.normal( dr, sdrw, n_samples), 0, sp)
        # the ending position at the prevsiout timestep t-1 is 
        # the initial of the current shift t, 
        evidence[ i] = np.cumsum( shifts)
        # find the first sample that run out of the bound (criterion)
        p = np.where( abs(evidence[i,]) > criterion)[0][0]
        # record the data 
        responses[ i] = np.sign( evidence[ i, p])
        latencies[ i] = int(p) 
    
    return responses, latencies, evidence
    
def plot_random_walks( latencies, evidence, 
                      criterion=3): 
    ## Plot up to 5 random-walk
    # creat timestep axis 
    n_reps = len(latencies)
    tbpn = np.min( [n_reps, 5])
    T = int(np.max(latencies[:tbpn]) + 20)
    timesteps = np.arange( T)
    _ = plt.figure(figsize=( 6, 4))
    for i in range(tbpn):
        data = evidence[ i, :int(latencies[i])+1]
        x = np.arange( 0, len( data))
        plt.plot( x, data, color=colors[i]) 
    plt.plot( timesteps, [criterion] * T, 'k--')
    plt.plot( timesteps, [-criterion] * T, 'k--')
    plt.xlabel( 'Time')
    plt.ylabel( 'Evidence')
    plt.xlim( [ 0, T]) 
    plt.ylim( [ -criterion-.5, criterion+.5])
    

def plot_reaction_time( responses, latencies):
    ## Show histogram
    _, axs = plt.subplots( 2, 1, figsize=( 7, 7))
    resps = [ 1, -1]
    for_title = [ 'Top', 'Bottom']
    for i, res in enumerate( resps):
        # get the index that match the response 
        ind = (responses == res)
        # get reaction time
        rt  = latencies[ ind]
        # turn into proption 
        prop = len( rt) / latencies.shape[0]     
        # get axes and show histogram
        ax = axs[i]
        x = np.arange(0,100*(1+np.max(latencies)//100),50)
        ax.hist( rt, bins=x, facecolor='gray', edgecolor='k')
        if i == 1:
            ax.set_xlabel( 'Decison time') 
        ax.set_ylabel( 'Frequency')   
        ax.set_title( f'{for_title[i]} response ({prop}), m={np.mean(rt):.3f}') 

    
if __name__ == '__main__':

    ##############################
    ##        Example 1         ##
    ##############################

    ## Simulate the behaviors for a basic random walk model
    responses, latencies, evidence = basic_random_walk_model()
    ## Visualize
    plot_random_walks( latencies, evidence)
    plt.savefig( f'{path}/fig1-random walks for a basic model.png', dpi=dpi)
    plot_reaction_time( responses, latencies)
    plt.savefig( f'{path}/fig2-RT for a basic model.png', dpi=dpi)

    ##############################
    ##        Example 2         ##
    ##############################

    ## Question: Does response time changes 
    #  when we change drift? No! see Fig3
    #  the portion of the samples reduces,
    #  but the mean of RTs stays the same. 
    drift = .03
    responses, latencies, evidence = basic_random_walk_model(drift=drift)
    ## Visualize
    plot_reaction_time( responses, latencies)
    plt.savefig( f'{path}/fig3-RT for a positive drift.png', dpi=dpi)

    ##############################
    ##        Example 3         ##
    ##############################

    ## Not every we start a trial with the same prior
    #  Add a jitter to the start point.  
    #  Fig4 tells that the error (bottom) takes less time.
    #  Reaching the bottom decision requires an unluck coincidence
    #  It happens when the start point (sp) is below 0. 
    drift = .035
    t2tsd = [ .8, .0]
    responses, latencies, evidence = t2t_variability_model(
                                        drift=drift, t2tsd=t2tsd, n_reps=1000)
    ## Visualize
    plot_reaction_time( responses, latencies)
    plt.savefig( f'{path}/fig4-RT for a trial-to-trial variability model.png', dpi=dpi)
    plot_random_walks( latencies, evidence)
    plt.savefig( f'{path}/fig4-S1-Find the higher start point.png', dpi=dpi)

    ##############################
    ##        Example 4         ##
    ##############################

    ## Not every shift from the previous trial is stable
    #  Add a jitter to the every drift.  
    #  Fig5 tells that the error (bottom) takes longer time.
    #  But the intuition of why the model behaviors is not
    #  clearly explained by the chapter
    drift = .03
    t2tsd = [ 0, .025]
    responses, latencies, evidence = t2t_variability_model(
                                        drift=drift, t2tsd=t2tsd, n_reps=1000)
    ## Visualize
    plot_reaction_time( responses, latencies)
    plt.savefig( f'{path}/fig5-RT for a trial-to-trial variability model.png', dpi=dpi)
    plot_random_walks( latencies, evidence)
    plt.savefig( f'{path}/fig5-S1-Find the higher start point.png', dpi=dpi)
