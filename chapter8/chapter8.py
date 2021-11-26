'''
Chapter 8: Bayesian Parameter Estimation

    @Zeming 

'''
import os
import numpy as np
import numpyro as npyro
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

#===================================
#       Simple Gibbs Sampler      
#===================================

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

#================================
#       Basic Functions      
#================================

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
    plot_samples( samples)
    plt.savefig( f'{path}/params_sumamry-{model_name}.png')

#===================================
#       Simple numpyro exmaple      
#===================================

def mymodel( obs):
    # priors: μ ~ Uni( -100, 100), σ ~ Uniform( 0, 100) 
    mu  = npyro.sample( 'mu', dist.Uniform( -100, 100))
    sig = npyro.sample( 'sig', dist.Uniform( 0, 100))
    tau = npyro.deterministic( 'tau', sig**(-2))
    # sample xx ~ N( μ, τ)
    xx = npyro.sample( 'x', dist.Normal( mu, tau), obs=obs)

#===================================
#       Signal detection model      
#===================================

def SD_model( obs, sigtrials=100, noistrials=100):
    # unpack obs
    h_obs, f_obs = obs 
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

#=====================================
#       Multinomial tree model       
#=====================================

def MTM( obs, sigtrials=100, noistrials=100):

    # upack observations
    hit_obs, fal_obs = obs 
    # θ1~beta(1,1), θ2~beta(1,1) 
    theta1 = npyro.sample( 'theta1', dist.Beta( 1, 1))
    theta2 = npyro.sample( 'theta2', dist.Beta( 1, 1))
    # p(h) = θ1 + (1-θ1)θ2, p(f) = θ2
    predh = npyro.deterministic( 'p(h)', 
                theta1 + ( 1 - theta1) * theta2)
    predf = npyro.deterministic( 'p(f)', theta2)
    # Nh ~ Binomial( p(h), N), Nf ~ Binomial( p(f), N)
    hit = npyro.sample( 'hit', dist.Binomial( sigtrials, predh),
                        obs=hit_obs)
    fal = npyro.sample( 'false', dist.Binomial( noistrials, predf),
                        obs=fal_obs)



if __name__ == '__main__':

    # Hand Gibbs
    gibbs_mvGauss()
    plt.savefig( f'{path}/Fig1_illustration_of_Gibbs_sampling', dpi=dpi)

    ## Gibbs using pyro 
    N = 1000
    obs = norm( loc=0, scale=2).rvs( N)
    pyro_sampling( obs, mymodel, 'simple', n_chains=1, n_warmup=2000)

    ## MCMC using pyro for Bayesian signal-detection model 
    obs = ( 60, 11)
    pyro_sampling( obs, SD_model, 'SDmodel')

    ## MCMC using pyro for multinomaila tree model  
    obs = ( 60, 11)
    pyro_sampling( obs, MTM, 'MTmodel')

