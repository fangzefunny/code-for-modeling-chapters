'''
Chapter 9: Bayesian Parameter Estimation

    @Zeming 

Note: due the implementation of gamma distribution, 
numpyro does not converge on this inference. 
An issue has been created in the numpyro github for solutions.
'''
import os
import numpy as np
import pandas as pd 
import numpyro as npyro
import numpyro.distributions as dist
import arviz as az
import matplotlib.pyplot as plt 
import seaborn as sns 

from jax import random 
import jax.numpy as jnp
from scipy.stats import norm, binom 
from numpyro.infer import MCMC, NUTS, Predictive

import warnings 
warnings.simplefilter( 'ignore',FutureWarning)

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=4 "
    "--xla_cpu_multi_thread_eigen=false "
)

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

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Basic functions     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def plot_samples(samples):
    lw = 4
    n_params = len( samples.keys())
    _, axs = plt.subplots( n_params, 2, figsize=( 5*2, 3*n_params))
    for i, key in enumerate(samples.keys()):
        ax = axs[ i, 0]
        if len( samples[key].shape) ==1:
            ax.plot( samples[key], color=Blue*.9, linewidth=lw)
        else:
            for j in range( samples[key].shape[1]):
                ax.plot( samples[key][ :, j], linewidth=lw/2)
        ax.set_xlabel( 'Iterations', fontsize=14)
        ax = axs[ i, 1]
        if len( samples[key].shape) ==1:
            sns.kdeplot( samples[key], ax=ax, color=Blue*.9, linewidth=lw)
        else:
            for j in range( samples[key].shape[1]):
                sns.kdeplot( samples[key][ :, j], ax=ax, linewidth=lw/2)
        ax.axvline( np.mean( samples[key]), color='gray', lw=1.5)
        ax.axvline( np.quantile( samples[key], .05), ls='--', color='gray')
        ax.axvline( np.quantile( samples[key], .95), ls='--', color='gray')
        ax.set_ylabel( 'Density', fontsize=14)
        ax.set_title( f'Density of {key}', fontsize=16)
    plt.tight_layout()

def pyro_sampling( args, obs, model, model_name, seed=0,
                 n_chains=1, n_samples=10000, n_warmup=30000):

    ## Fix the random seed
    rng_key = random.PRNGKey(seed)

    ## Sampling 
    kernel = NUTS( model)
    posterior = MCMC( kernel, num_chains=n_chains, 
                              num_samples=n_samples, 
                              num_warmup=n_warmup,)
    posterior.run( rng_key, args, obs)
    samples = posterior.get_samples()
    
    ## Summarize the sampling results
    print( posterior.print_summary())
    az.plot_trace( az.from_numpyro( posterior), compact=True)
    plt.tight_layout()
    plt.savefig( f'{path}/{model_name}-params sumamry.png', dpi=dpi)

    return samples

#==============================================
#     Hierarchical Signal Detection Model    
#==============================================

def sim_hSDM():

    ## Prepare the data 
    args = dict()
    args[ 'n_subj'] = 10
    args[ 'sigtrials'] = args['noistrials'] = 100
    h_obs = binom( n=args[ 'sigtrials'], p=.8).rvs( 10)
    f_obs= binom( n=args['noistrials'], p=.2).rvs( 10)

    ## Sample to get the results
    posterior = pyro_sampling( args, ( h_obs, f_obs), hSDM, 'hSDM')

    ## Show performance

def hSDM( args, obs=None, eps=.001):

    if obs:
        h_obs, f_obs = obs[0], obs[1]
    else:
        h_obs, f_obs = None, None 

    ## Population level distributions
    # μd ~ N( 0, eps), μb ~ N( 0, eps), 
    # τb ~ Gamma( eps, eps), τb ~ Gamma( eps, eps),
    mud = npyro.sample( 'μ_d', dist.Normal( 0, eps))
    mub = npyro.sample( 'μ_b', dist.Normal( 0, eps))
    taud = npyro.sample( 'τ_d', dist.Gamma( eps, eps))
    taub = npyro.sample( 'τ_b', dist.Gamma( eps, eps))

    ## Individual level 
    with npyro.plate( 'plate_i', args[ 'n_subj']):

        # split data 
        #hi_obs, fi_obs = obs[ 'h_obs'][ ind], obs[ 'f_obs'][ind]

        # prior d ~ N(μd,τd), b~N(μb,τb)
        d = npyro.sample( 'd', dist.Normal( mud, taud))
        b = npyro.sample( 'b', dist.Normal( mub, taub))

        # area under curves p(h) = Φ( d/2-b), p(f) = Φ( -d/2-b)
        phi_hit = npyro.deterministic( 
                    'p_hit', dist.Normal( 0, 1).cdf( d/2-b))
        phi_fal = npyro.deterministic( 
                    'p_fal', dist.Normal( 0, 1).cdf( -d/2-b))

        # observed hit ~ Bern(sig; p(h)); false ~ Bern( noise; p(f))
        npyro.sample( 'h_times', dist.Binomial( args[ 'sigtrials'], probs=phi_hit), 
                        obs=h_obs)
        npyro.sample( 'f_times', dist.Binomial( args['noistrials'], probs=phi_fal), 
                        obs=f_obs)

def show_hit_false( obs, samples, seed=1234):

    ## Check goodness of fit 
    pred_data = []
    for i in range( obs['n_subj']):
        df = pd.DataFrame(columns=['subj_id'])
        df['subj_id'] = i
        pred_data.append(df)
    pred_data = pd.concat( pred_data, ignore_index=True)

    pred = Predictive( hSDM, samples, return_sites=['h_times', 'f_times'])
    pred_samples = pred( random.PRNGKey( seed), obs)
    pred_data[ 'h_pred'] = pred_samples['h_times'][ 0, :]/ obs['sigtrials']
    pred_data[ 'f_pred'] = pred_samples['f_times'][ 0, :] / obs['noistrials']
    pred_data[ 'h_obs']  = obs[ 'h_obs'] / obs['sigtrials']
    pred_data[ 'f_obs']  = obs[ 'f_obs'] / obs['noistrials']

    nc = 2 
    _, axs = plt.subplots( 1, nc, figsize=( 3.5*nc, 3.5))
    ax = axs[ 0]
    ax.scatter( pred_data[ 'h_obs'], pred_data[ 'h_pred'],
                edgecolor=Red, facecolor='None', s=50)
    ax.plot( np.linspace( 0, 1, 10), np.linspace( 0, 1, 10),
                color='gray', ls='--')
    ax.set_ylabel( 'Human obs', fontsize=14)
    ax.set_ylabel( 'Model pred', fontsize=14)
    ax.set_title( f'Hits', fontsize=16)

    ax = axs[ 1]
    ax.scatter( pred_data[ 'f_obs'], pred_data[ 'f_pred'],
                edgecolor=Red, facecolor='None', s=50)
    ax.plot( np.linspace( 0, 1, 10), np.linspace( 0, 1, 10),
                color='gray', ls='--')
    ax.set_ylabel( 'Human obs', fontsize=14)
    ax.set_ylabel( 'Model pred', fontsize=14)
    ax.set_title( f'Falses', fontsize=16)    
    plt.tight_layout()
    plt.savefig( f'{path}/hSDM-goodness of fit.png', dpi=dpi)

#================================================
#     Hierarchical Modeling of Forgetting     
#================================================

def sim_hMF( seed=1234):

    ## Generate data
    rng = np.random.RandomState( seed)
    tlags = [ 0, 1, 5, 10, 20, 50]
    n_subj, T, n_items = 4, len(tlags), 20
    recalls = np.zeros( [ n_subj, T]) + np.nan 
    for sub_i in range(n_subj):
        a = .2*rng.rand()
        b = .1*rng.rand() + .9
        alpha = .3*rng.rand() + .1
        print( f'Sub {sub_i}: a={a:.2f}, b={b:.2f}, alpha={alpha:.2f}')
        for j, t in enumerate(tlags):
            p = a + (1-a) * b * np.exp(-alpha*t)
            recalls[ sub_i, j] = binom( n=n_items, p=p).rvs()
    args = { 'n_subj': n_subj, 'n_items': 20, 'T': T}

    ## Sample to estimate
    posterior = pyro_sampling( args, recalls, hMF, 'hMF', n_samples=5000, n_warmup=20000)


def hMF( args, obs=None, eps=5):

    ## Populational level distributions
    mu_alpha  = npyro.sample( 'mu_alpha', dist.Uniform( 0, 1))
    tau_alpha = npyro.sample( 'tau_alpha', dist.Uniform( 0, eps))
    mu_a      = npyro.sample( 'mu_a', dist.Uniform( 0, 1))
    tau_a     = npyro.sample( 'tau_a', dist.Uniform( 0, eps))
    mu_b      = npyro.sample( 'mu_b', dist.Uniform( 0, 1))
    tau_b     = npyro.sample( 'tau_b', dist.Uniform( 0, eps))

    ## Individual distributions
    alpha = npyro.sample( 'alpha', dist.TruncatedDistribution(
                    dist.Normal( jnp.broadcast_to( mu_alpha, [ args['n_subj'], 1]), 
                    tau_alpha), low=0, high=1))
    a     = npyro.sample( 'a', dist.TruncatedDistribution( 
                    dist.Normal( jnp.broadcast_to( mu_a, [ args['n_subj'], 1]), 
                    tau_a), low=0, high=1))
    b     = npyro.sample( 'b', dist.TruncatedDistribution( 
                    dist.Normal( jnp.broadcast_to( mu_b, [ args['n_subj'], 1]), 
                    tau_b), low=0, high=1))

    ## Each time steps
    sub_id = np.arange( args['n_subj'])
    t_lags = np.arange( args['T']).reshape([1,-1])
    theta = npyro.deterministic( 'theta', a[sub_id] + (1 - a[sub_id]
                ) * b[sub_id] * jnp.exp(-alpha[sub_id] * t_lags))
    
    return npyro.sample( 'recall', dist.Binomial( 
                        args['n_items'], probs=theta), obs=obs)

def viz_exp_pow_fns( ):

    # define variables 
    t_lags = np.linspace( 0, 40, 40)
    alphas = np.logspace( np.log(0.1), np.log(1), 5)
    betas  = np.logspace( np.log(0.1), np.log(1), 5)
    trans  = np.linspace( .0, .7, 5) + .3

    # define functions 
    exp_fn = lambda t, a: np.exp( -a * t)
    pow_fn = lambda t, b: ( 1 + t) ** (-b) 

    # get data 
    exp_data = np.vstack( [ list(map( exp_fn, t_lags, [alpha]*len(t_lags)
                        )) for alpha in alphas])
    pow_data = np.vstack( [ list(map( pow_fn, t_lags, [beta]*len(t_lags)
                        )) for beta in betas])

    # visualization
    nc = 2 
    _, axs = plt.subplots( 1, nc, figsize=( 3.5*nc, 3.5))
    ax = axs[0]
    for i in range(len(alphas)):
        ax.plot( t_lags, exp_data[ i, :], color=Blue, alpha=trans[i])
    ax.set_xlabel( 'delays', fontsize=13)
    ax.set_ylabel( 'decay rate', fontsize=13)
    ax.set_title( 'Exponential function', fontsize=15)
    str_alpha = r'$\alpha$'
    ax.legend( [ f'{str_alpha}={a:.2f}' for a in alphas])
    ax.set_ylim( [ -.05, 1.05])

    ax = axs[1]
    for i in range(len(betas)):
        ax.plot( t_lags, pow_data[ i, :], color=Red, alpha=trans[i])
    ax.set_xlabel( 'delays', fontsize=13)
    ax.set_ylabel( 'decay rate', fontsize=13)
    ax.set_title( 'Power function', fontsize=15)
    str_beta = r'$\beta$'
    ax.legend( [ f'{str_beta}={b:.2f}' for b in betas])
    ax.set_ylim( [ -.05, 1.05])
    plt.tight_layout()
    plt.savefig( f'{path}/hMF-exp pow fns.png', dpi=dpi)

if __name__ == '__main__':

    ## Hierarchical signal-detection model
    sim_hSDM()


    ## Hierarchical modeling of forgetting
    viz_exp_pow_fns( )
    sim_hMF()
    


