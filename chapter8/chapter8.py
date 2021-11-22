'''
Chapter 8: Bayesian Parameter Estimation

    @Zeming 

'''
import os
from typing import no_type_check
import numpy as np
import numpyro as npyro
from numpyro.diagnostics import gelman_rubin
import numpyro.distributions as dist
import matplotlib.pyplot as plt 
import seaborn as sns 

from jax import random 
from scipy.stats import multivariate_normal, norm
from numpyro.infer import MCMC, NUTS

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
Section 8.1 Simple Gibbs Sampler
'''

def gibbs_mvGauss( n_samples=1000, seed=2021):
    
    ## Fix random seed
    np.random.seed( seed)

    ## Construct a multi-variate Gauss
    rho = .8
    mux = muy = 0
    sigx, sigy = 1, .5
    cov = np.array([ sigx**2, rho*sigx*sigy, 
                    rho*sigx*sigy, sigy**2]).reshape([ 2, 2])
    
    ## Get the ground truth 
    x, y = np.mgrid[ -3:3:.1, -3:3:.1]
    z = multivariate_normal( ( mux, muy), cov
                    ).pdf( np.dstack( ( x, y)))
    
    ## Get sample from Gibbs sampler
    sig_x1y = np.sqrt( sigx**2 * ( 1 - rho**2)) # σ_y|x = sqrt( σ^2_x( 1 - rho^2)
    sig_y1x = np.sqrt( sigy**2 * ( 1 - rho**2)) # σ_x|y = sqrt( σ^2_x( 1 - rho^2)
    rxy, ryx = rho*( sigx/sigy), rho*( sigy/sigx)
    chain = [ [ -2, 2]]
    done = False 
    while not done:
        y_t = chain[-1][1]
        x_next = norm( rxy*y_t, sig_x1y).rvs()
        y_next = norm( ryx*x_next, sig_y1x).rvs()
        chain.append( [ x_next, y_next])
        # check termination
        if len(chain) >=n_samples: done = True

    ## Visualize
    fig, axs = plt.subplots( 1, 2, figsize=( 8, 4))
    ax = axs[ 0]
    ax.contour( x, y, z)
    ax.set_xlim( [ -3, 3])
    ax.set_ylim( [ -3, 3])
    ax = axs[ 1]
    ax.contour( x, y, z)
    for samp in chain[ 500:]:
        ax.scatter( samp[0], samp[1], color=Red, 
                    edgecolor='gray', s=10)
    ax.set_xlim( [ -3, 3])
    ax.set_ylim( [ -3, 3])

def mymodel( obs):
    # priors: μ ~ Uni( -100, 100), σ ~ Uniform( 0, 100) 
    mu  = npyro.sample( 'mu', dist.Uniform( -100, 100))
    sig = npyro.sample( 'sig', dist.Uniform( 0, 100))
    tau = npyro.deterministic( 'tau', sig**(-2))
    # sample xx ~ N( μ, τ)
    npyro.sample( 'x', dist.Normal( mu, tau), obs=obs)

def plot_samples(samples):
    lw = 4
    n_params = len( samples.keys())
    _, axs = plt.subplots( n_params, 2, figsize=( 5*2, 3*n_params))
    for i, key in enumerate(samples.keys()):
        ax = axs[ i, 0]
        ax.plot( samples[key], color=Blue*.9, linewidth=lw)
        ax.set_xlabel( 'Iterations', fontsize=14)
        ax = axs[ i, 1]
        sns.kdeplot( samples[key], ax=ax, color=Blue*.9, linewidth=lw)
        ax.axvline( np.mean( samples[key]), color='gray', lw=1.5)
        ax.axvline( np.quantile( samples[key], .05), ls='--', color='gray')
        ax.axvline( np.quantile( samples[key], .95), ls='--', color='gray')
        ax.set_ylabel( 'Density', fontsize=14)
        ax.set_title( f'Density of {key}', fontsize=16)
    plt.tight_layout()

    
def pyro_sampler( model, model_name):

    ## Get obs
    N = 1000
    x = norm( loc=0, scale=2).rvs( N)
    rng_key = random.PRNGKey(1234)
    
    ## Sampling 
    kernel = NUTS( model)
    posterior = MCMC( kernel, num_samples=5000, num_warmup=10000)
    posterior.run( rng_key, x)
    print( posterior.print_summary())
    samples = posterior.get_samples()
    plot_samples(samples)
    plt.savefig( f'{path}/params_sumamry-{model_name}.png')

def SD_model( h_obs, f_obs, sigtrials, noistrials,):
    # prior d ~ N(1,1), b~N(0,1)
    d = npyro.sample( 'discrim', dist.Normal( 1, 1))
    b = npyro.sample( 'bias', dist.Normal( 0, 1))
    # area under curves p(h) = Φ( d/2-b), p(f) = Φ( -d/2-b)
    phi_hit = npyro.deterministic( 
                'p_hit', dist.Normal( 0, 1).cdf( d/2-b))
    phi_false = npyro.deterministic( 
                'p_false', dist.Normal( 0, 1).cdf( -d/2-b))
    # observed hit ~ Bern(sig; p(h)); false ~ Bern( noise; p(f))
    hit = npyro.sample( 'hit', dist.Binomial( sigtrials, phi_hit), obs=h_obs)
    false = npyro.sample( 'false', dist.Binomial( noistrials, phi_false), obs=f_obs)

def SD_sampler( model, model_name):

    ## Get obs
    h_obs, f_obs = 60, 11
    sigtrials = noistrials = 100
    rng_key = random.PRNGKey(1234)
    
    ## Sampling 
    kernel = NUTS( model)
    posterior = MCMC( kernel, num_chains=4, num_samples=5000, num_warmup=100000)
    posterior.run( rng_key, h_obs, f_obs, sigtrials, noistrials)
    print( posterior.print_summary())
    samples = posterior.get_samples()
    plot_samples( samples)
    plt.savefig( f'{path}/params_sumamry-{model_name}.png')


if __name__ == '__main__':

    ## Hand Gibbs
    # gibbs_mvGauss()
    # plt.savefig( f'{path}/Fig1_illustration_of_Gibbs_sampling', dpi=dpi)

    ## Gibbs using pyro 
    pyro_sampler( mymodel, 'simple')

    ## MCMC using pyro for Bayesian signal-detection model 
    SD_sampler( SD_model, 'SDmodel')

