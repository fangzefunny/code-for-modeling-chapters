'''
Chapter 9: Bayesian Parameter Estimation

    @Zeming 

'''
import os
from typing import no_type_check
import numpy as np
import pandas as pd 
import pyro
import torch 
import numpyro as npyro
import numpyro.distributions as dist
import arviz as az
import matplotlib.pyplot as plt 
import seaborn as sns 

from jax import random 
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

def pyro_sampling( obs, model, model_name, seed=1234,
                 n_chains=4, n_samples=5000, n_warmup=10000):

    ## Fix the random seed
    rng_key = random.PRNGKey(seed)

    ## Sampling 
    kernel = NUTS( model)
    posterior = MCMC( kernel, num_chains=n_chains, 
                              num_samples=n_samples, 
                              num_warmup=n_warmup,)
    posterior.run( rng_key, obs)
    samples = posterior.get_samples()
    
    ## Summarize the sampling results
    print( posterior.print_summary())
    az.plot_trace( az.from_numpyro( posterior), compact=True)
    plt.savefig( f'{path}/{model_name}-params sumamry.png', dpi=dpi)

    return samples

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Hierarchical Signal Detection Model     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def hSDM( obs, eps=.001):

    ## Population level distributions
    # μd ~ N( 0, eps), μb ~ N( 0, eps), 
    # τb ~ Gamma( eps, eps), τb ~ Gamma( eps, eps),
    mud = npyro.sample( 'μ_d', dist.Normal( 0, eps))
    mub = npyro.sample( 'μ_b', dist.Normal( 0, eps))
    taud = npyro.sample( 'τ_d', dist.Gamma( eps, eps))
    taub = npyro.sample( 'τ_b', dist.Gamma( eps, eps))

    ## Individual level 
    with npyro.plate( 'plate_i', obs[ 'n_subj']):

        # split data 
        #hi_obs, fi_obs = obs[ 'h_obs'][ ind], obs[ 'f_obs'][ind]

        # prior d ~ N(μd,τd), b~N(μb,τb)
        d = npyro.sample( 'd', dist.Normal( mud, taud))
        b = npyro.sample( 'b', dist.Normal( mub, taub))

        # area under curves p(h) = Φ( d/2-b), p(f) = Φ( -d/2-b)
        phi_hit = npyro.deterministic( 
                    'p_hit', dist.Normal( 0, 1).cdf( d/2-b))
        phi_false = npyro.deterministic( 
                    'p_false', dist.Normal( 0, 1).cdf( -d/2-b))

        # observed hit ~ Bern(sig; p(h)); false ~ Bern( noise; p(f))
        h = npyro.sample( 'h_times', dist.Binomial( obs[ 'sigtrials'], phi_hit), 
                        obs=obs[ 'h_obs'])
        f = npyro.sample( 'f_times', dist.Binomial( obs['noistrials'], phi_false), 
                        obs=obs[ 'f_obs'])
        return h, f


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

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Hierarchical Modeling of Forgetting     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def HMF( obs, eps=.001):

    ## Populational level distributions
    mu_alpha  = npyro.sample( 'mu_alpha', dist.Uniform( 0, 1))
    tau_alpha = npyro.sample( 'tau_alpha', dist.Gamma( eps, eps))
    mu_a      = npyro.sample( 'mu_a', dist.Uniform( 0, 1))
    tau_a     = npyro.sample( 'tau_b', dist.Gamma( eps, eps))
    mu_b      = npyro.sample( 'mu_b', dist.Uniform( 0, 1))
    tau_b     = npyro.sample( 'tau_b', dist.Gamma( eps, eps))

    ## Individual distributions
    with npyro.plate( 'plate_i', obs[ 'n_subj']):
        alpha = npyro.sample( 'alpha', np.clip( 
            dist.Normal( mu_alpha, tau_alpha), 0, 1))
        a = npyro.sample( 'a', np.clip( 
            dist.Normal( mu_a, tau_a), 0, 1))
        b = npyro.sample( 'b', np.clip( 
            dist.Normal( mu_b, tau_b), 0, 1)) 

        def trans_fn( t, t_obs):
            theta = npyro.deterministic( 'theta',
                a + ( 1- a) * b * np.exp( -alpha * t))
            np.pyro.sample( 'k', dist.Binomial( theta, obs['n_items']),
                        obs=t_obs)
            return t+1, None

        with npyro.plate( 'plate_j', obs[ 'T']):
            pass 
            



if __name__ == '__main__':

    ## Hand Gibbs
    # gibbs_mvGauss()
    # plt.savefig( f'{path}/Fig1_illustration_of_Gibbs_sampling', dpi=dpi)

    ## Gibbs using pyro 
    obs = dict()
    obs[ 'n_subj'] = 10
    obs[ 'sigtrials'] = obs['noistrials'] = 100
    obs[ 'h_obs'] = binom( n=obs[ 'sigtrials'], p=.8).rvs( 10)
    obs[ 'f_obs'] = binom( n=obs['noistrials'], p=.2).rvs( 10)
    posterior = pyro_sampling( obs, hSDM, 'hSDM')
    show_hit_false( obs, posterior)


