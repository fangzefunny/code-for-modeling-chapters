'''
Chapter 7: Bayesian Parameter Estimation

    @Zeming 

'''
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from scipy.stats import norm, vonmises
from scipy.special import i1e, i0e


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
dpi = 250

def MH_samp( chain_len=5000, init=150, bernin_len=200, 
             obs=144, propsd=2, seed=122021):
    '''Metropolis-Hastings Sampling
    
    Estimate a normal distribution
    '''
    ## Init the chain
    chain = [init,]
    t = 0
    done = False 
    np.random.seed(seed)

    ## Start sampling
    while not done:
        current = chain[-1]  # θt
        p_D1curr = norm( current, 15).pdf( obs)
        proposal = current + norm( 0, propsd).rvs() # θt+1 ~ q(θt+1|θt)
        p_D1prop = norm( proposal, 15).pdf( obs) # p(D|θt+1)
        # reject with probability min( 1, p(D|θt+1) / p(D|θt))
        alpha = np.min( [ 1, p_D1prop / p_D1curr])
        accept = np.random.rand() <  alpha
        sample = proposal if accept else current
        chain.append( sample)    
        t += 1
        # check termination
        if t >= chain_len + bernin_len:
            done = True  

    return chain[ bernin_len+1:], chain

def MH_samp_prior( prior, chain_len=5000, init=150, bernin_len=200, 
                obs=144, obssd=20, propsd=2, seed=122021):
    '''Metropolis-Hastings Sampling
    
    Estimate a normal distribution
    '''
    ## Init the chain
    chain = [init,]
    t = 0
    done = False 
    np.random.seed(seed)

    ## Start sampling
    while not done:
        current = chain[-1]  # θt
        f_D_curr = norm( current, obssd).pdf( obs) \
                    * prior.pdf( current) # p(D|θt)p(θt)
        proposal = current + norm( 0, propsd).rvs() # θt+1 ~ q(θt+1|θt)
        f_D_prop = norm( proposal, obssd).pdf( obs) \
                    * prior.pdf( proposal) # p(D|θt+1)p(θt+1)
        # reject with probability min( 1, f(D,θt+1) / f(D,θt))
        alpha = np.min( [ 1, f_D_prop / f_D_curr])
        accept = np.random.rand() <  alpha
        sample = proposal if accept else current
        chain.append( sample)    
        t += 1
        # check termination
        if t >= chain_len + bernin_len:
            done = True  

    return chain[ bernin_len+1:], chain

def plot_param( chains, mode='param_est', prior=None,
                t_dist=( 150, 15),):

    ## Upack the input 
    chain, chain_wbernin, = chains
    bernin_len = len( chain_wbernin) - len( chain)
    ## Create ax axis 
    n_bins = 80
    x = np.linspace( t_dist[0] * .4, t_dist[0] * 1.6, n_bins)
    ## Make figure 
    plt.figure( figsize=( 5, 4))
    sns.set_style("whitegrid", {'axes.grid' : False})
    if mode == 'param_est':   
        lg = ['Normal pdf', 'All MCMC', 'Excluding burnin']
        plt.plot( x, norm( t_dist[0], t_dist[1]).pdf(x), color=Red)
        sns.kdeplot( np.array( chain), bw_adjust=1.)
        sns.kdeplot( np.array( chain_wbernin), bw_adjust=1., color='k', linestyle='--')
        if prior:
            plt.plot( x, prior.pdf(x), ':', color=Blue)
            lg.append( 'Prior PDF')
        plt.ylabel( 'Density', fontsize=13)
        plt.xlim( [ t_dist[0] * .4, t_dist[0] * 1.6])
        plt.xlabel( 'Sampled values of $\mu$')
        plt.legend( lg)
        
    elif mode == 'accept_val':
        plt.plot( range( bernin_len), chain_wbernin[:bernin_len], 
                     color='gray')
        plt.plot( range( bernin_len, len(chain_wbernin)),
                    chain_wbernin[bernin_len:], 
                    color='k')
        plt.ylabel( 'Val. of accepted sample', fontsize=13)
        plt.xlabel( 'Iteration')
        plt.legend( ['bernin', 'samples'])
    plt.tight_layout()

'''
################################################
##    Section 7.1.2 Mixature model Example    ##
################################################
'''

def plot_mixature_model():
    '''Visualize zhang and luck 08, mixature model 
    '''
    ## Load data 
    data = pd.read_table( f'{path}/zhl08subj1.dat', sep=',', header=None)
    data.columns = [ 'setsize', 'errors']

    ## The fit target is aggregated over subjects using
    #  vincent averaging: Get ground turth 
    data['errdiscrete'] = pd.cut( data['errors'], 
                            bins=np.arange( -180, 180+20, 20),
                            labels=False)
    npss = np.mean(data['setsize'].value_counts().values)
    vwmmeans = data.groupby(['setsize', 'errdiscrete']).agg( 
                    {'errors': lambda x: len(x)/npss}
                    ).reset_index().sort_values(['errdiscrete'])

    ## Estimate parameters for each set size 
    svalues = [ .5, 20]
    preds, posteriors = [], []
    ssz = [ 3, 6]
    for sz in ssz:
        ind = (data['setsize'] == sz)
        sub_data = data['errors'].values[ ind]
        cp = getMixtmodel( sub_data, svalues)
        preds.append( cp['pred'])
        posteriors.append( cp['posterior'])
    
    ## Now plot the results
    # show the raw datas
    x4preds = np.linspace( 0, 20, len(preds[0]))
    plt.figure(figsize=( 6, 6))
    colors = [ Blue, Red]
    linestyles = [ '-', '--']
    for i, sz in enumerate(ssz):
        p_true = vwmmeans['errors'][ vwmmeans['setsize']==sz].values
        x_true = vwmmeans['errdiscrete'][ vwmmeans['setsize']==sz].values
        plt.scatter( x_true, p_true, s=60,
                edgecolor=colors[i], facecolor='None', label=f'Human set size={sz}')
        plt.plot( x4preds, preds[i] * len(x4preds)/len(x_true), linestyles[i], 
                color='k', label=f'Model set size={sz}')
    plt.xticks( range( 0, 20, 2), np.arange( -180, 220, 40))
    plt.xlabel( 'Difference from actual color values (deg)')
    plt.ylabel( 'Porportion of responses')
    plt.legend( ['Human set size=3', 'Human set size=6',
                 'Model set size=3', 'Model set size=6'])
    plt.savefig( f'{path}/Fig3_MixtureModel_Responses.png', dpi=dpi)
   
    # show posterior  
    nr = len( ssz)
    nc = posteriors[0].shape[0]
    param_name = [ 'g', '$\sigma_{vm}$']
    fig, axs = plt.subplots( nr, nc, figsize=( nc*4, nr*3.4))
    for i in range( nr):
        for j in range( nc):
            ax = axs[ i, j]
            sns.kdeplot( posteriors[i][j], ax=ax,
                bw_adjust=1., color=Red, lw=2)
            ax.axvline( np.mean( posteriors[i][j]), color='gray', lw=1.5)
            ax.axvline( np.quantile( posteriors[i][j], .05), ls='--')
            ax.axvline( np.quantile( posteriors[i][j], .95), ls='--')
            ax.set_yticks( [])
            ax.set_ylabel( 'Density', fontsize=14)
            ax.set_xlabel( f'sampled value of {param_name[j]}'
                            , fontsize=14)
    plt.tight_layout()
    plt.savefig( f'{path}/Fig4_MixtureModel_posterior.png', dpi=dpi)
            
def getMixtmodel( data, svalues, bnds=[ ( 0, 1), ( 4, 360)],
                  chain_len=5000, burnin_len=1000, seed=1234):
    '''
    params:
        g: lapse
        sdv: standard devation 
    '''
    
    ## Set random generator 
    rng = np.random.RandomState( seed)

    ## Define some values
    propsd = np.array(svalues) * .05
    lb = np.array( [ bnd[0] for bnd in bnds])
    ub = np.array( [ bnd[1] for bnd in bnds])

    ## Start sampling
    chain, done = [svalues], False
    while not done:
        # get current sample and propose a sample
        curr, doitagain = chain[-1], True
        while doitagain:
            prop = curr + norm( 0, propsd).rvs(2)
            doitagain = any(prop < lb) or any( prop > ub)
        # llh( prop), llh( curr), ratio
        like_curr = logmixturepdf( data, curr[0], curr[1]) +\
                        logprior( curr[0], curr[1])
        like_prop = logmixturepdf( data, prop[0], prop[1]) +\
                        logprior( prop[0], prop[1])
        lratio    = np.exp( like_prop - like_curr)
        sample = prop if rng.rand() < lratio else curr
        chain.append( sample)
        # check termination 
        if len(chain) >= chain_len+burnin_len: done = True 
    chain = np.vstack( chain).T  #  2 x chain_len
    finparam = np.mean( chain, axis=1)
    print( finparam)

    td = np.arange( -180, 181)
    pred = mixturepdf( td, finparam[0], finparam[1])
    pred /= np.sum( pred) # normalize 
    posterior = chain[ :, burnin_len:]
    return {'pred': pred, 'posterior': posterior}            

def logmixturepdf( data, g, sdv):
    '''VonMix distribution with lapse
    '''
    p_r1x_e= mixturepdf( data, g, sdv)
    return np.sum(np.log( p_r1x_e + 1e-18))

def mixturepdf( data, g, sdv):
    data_deg = np.pi * data / 180
    p_r1x = vonmises( sd2k(sdv)).pdf( data_deg)
    return (1-g)*p_r1x + g*1/360

def logprior( g, sdv):
    p_g   = jp4kappa( sd2k( sdv)) 
    p_sdv = jp4prop( g) 
    return np.log( p_g + 1e-18) + np.log( p_sdv)

def sd2k( d):
    '''Calculate kappa from data std
    '''
    s = np.pi * d / 180 
    r = np.exp( -s**2/2)
    k = 1/ ( r**3 - 4*r**2 + 3*r)
    if r < .85: k = -.4 + 1.39*r + .43/(1-r)
    if r < .53: k = 2*r + r**3 + 5*r**5/6
    return k

def jp4kappa( k):
    z = np.exp( (np.log( i1e( k))+ k) - 
                (np.log( i0e( k)) + k))
    return z * ( k - z - k*z**2)
    
jp4prop = lambda p: p**(-.5) * (1-p)**(-.5)


'''
##############################################
##    Section 7.2.3 ABC SDmodel Example     ##
##############################################
'''

def ABC_est( y=np.array( [60, 11]), prior_params=[ (1,0), (1,1)], n_trials=100,
             epsilon=1, n_samples=1000):

    ## Define some variables 
    dmu, bmu = prior_params[0][0], prior_params[0][1]
    dsig, bsig = prior_params[1][0], prior_params[1][1]
    
    ## Start simulations
    posterior, done, t = [], False, 0
    while not done:
        p = '.' * int( 40 * t / n_samples)
        print( f'{p}{t}')
        # sample a p( θ, data)
        while True:
            dprop = norm( dmu, dsig).rvs()
            bprop = norm( bmu, bsig).rvs()
            x = simsdt( dprop, bprop, n_trials)
            if np.sqrt( np.sum( (x - y)**2) <= epsilon): break
        posterior.append( [ dprop, bprop])
        # check termination
        if t >= n_samples: done = True
        t += 1
    ## Show estimation quality 
    print(np.mean( np.vstack( posterior), axis=0))

def simsdt( d, b, n_trials):
    # x_o
    old = norm( d, 1).rvs( int(n_trials/2))
    # ∫^{d/2} N(x_o|0,b) x_o dx_0
    hits = np.sum( old > ( d/2 + b)) / (n_trials/2) * 100
    # x_n
    new = norm( 0, 1).rvs( int(n_trials/2))
    fas = np.sum( new > ( d/2 + b)) / (n_trials/2) * 100
    return np.array( [ hits, fas])

if __name__ == '__main__':

    ## Benchmark 
    chain = MH_samp()
    plot_param( chain)
    plt.savefig( f'{path}/Fig1A_MCMC_output.png', dpi=dpi)
    plot_param( chain, mode='accept_val')
    plt.savefig( f'{path}/Fig1A_MCMC_accepted_val.png', dpi=dpi)

    ## Biased init, small sample 
    chain = MH_samp( chain_len=500, init=10)
    plot_param( chain)
    plt.savefig( f'{path}/Fig1B_MCMC_output.png', dpi=dpi)
    plot_param( chain, mode='accept_val')
    plt.savefig( f'{path}/Fig1B_MCMC_accepted_val.png', dpi=dpi)

    ## Similar proposal distribution 
    chain = MH_samp( chain_len=5000, init=500, propsd=20)
    plot_param( chain)
    plt.savefig( f'{path}/Fig1C_MCMC_output.png', dpi=dpi)
    plot_param( chain, mode='accept_val')
    plt.savefig( f'{path}/Fig1C_MCMC_accepted_val.png', dpi=dpi)

    ## Add prior1  
    prior = norm( 328, 88)
    chain = MH_samp_prior( prior, obs=415, obssd=20, 
                init=500, chain_len=8000, bernin_len=500)
    plot_param( chain, t_dist=( 415, 20), prior=prior)
    plt.savefig( f'{path}/Fig2A_MCMC_prior1.png', dpi=dpi)

    ## Add prior2 
    prior = norm( 328, 20)
    chain = MH_samp_prior( prior, obs=415, obssd=20, 
                init=500, chain_len=8000, bernin_len=500)
    plot_param( chain, t_dist=( 415, 20), prior=prior)
    plt.savefig( f'{path}/Fig2B_MCMC_prior2.png', dpi=dpi)
    
    ## plot mixature model 
    plot_mixature_model()

    ## ABC method
    ABC_est()
    
    