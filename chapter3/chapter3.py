'''
Chapter 3: Basic Parameter Estimation Techniques

    @Zeming 

'''
import os
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin 


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

def synthesize_data( rho=.8, intercept=0., n_data_pts=20):

    ## Generate synthetic data
    data = np.zeros( [ n_data_pts , 2])
    # sythesize the predictor data x 
    data[ :, 1] = np.random.normal( size=n_data_pts)
    # sythesize the predictor data y
    data[ :, 0] = np.random.normal( size=n_data_pts) * np.sqrt( 1 - rho ** 2) \
                                   + data[ :, 1] * rho + intercept
    return data 

def fit_reg_params( data):

    ## do convential regression analysis 
    reg = LinearRegression()
    # get the initialized parameters
    coef0 = [ -1., .2]

    # obtain the fitted parameters
    reg.fit( data[ :, [1]], data[ :, 0])
    coef_fit = list(reg.coef_) + [reg.intercept_]

    return coef0, coef_fit

def plot_reg( data, params):
    
    ## uppack input 
    x, y = data[ :, 1], data[ :, 0]
    names = [ 'init', 'fit']

    ## Visualize the model behavior
    _, axs = plt.subplots( 1, 2, figsize=( 8, 4)) 
    for i, coef in enumerate( params):
        # get axes
        ax = axs[ i]
        b1, b0 = coef 
        y_hat = b0 + b1 * x
        # scatter plot to show the data point
        ax.scatter( x, y, 
                    s=60, facecolors='none', edgecolors=Red) 
        ax.plot( x, y_hat, color='k')
        ax.set_xlim( [ -2, 2])
        ax.set_ylim( [ -2, 2])
        ax.set_xlabel( 'X')
        ax.set_ylabel( 'Y')
        ax.set_title( f'{names[i]} RMSE = { np.sqrt(mean_squared_error( y, y_hat)):.4f}')

def powediscrep( params, rec, ri):
    '''Discrepancy (Loss function) for power forgetting model

    '''
    # check if the params within 0，1
    if (np.min( params) < 0) or ( np.max(params) > 1):
        return 1e6
    else: 
        # make prediction 
        pow_pred = params[0] * ( params[1] * ri + 1) ** (-params[2])
        return np.sqrt( np.sum( (pow_pred - rec) ** 2) / len( ri)) 

def forget_model():
    '''

    '''
    ## Get the human behavioral data from Carpenter et al 2008 exp1
    rec = np.array([ .93, .88, .86, .66, .47, .34])  # reaction 
    ri  = np.array([ .0035, 1, 2, 7, 14, 42])        # time interval 

    ## Search for the optimal parameter
    sparams = [ 1, .05, .7]
    params_opt = fmin( powediscrep, sparams, args=( rec, ri), disp=False)
    x_ri = np.arange( 0, np.max(ri)+2)
    pow_pred = params_opt[0] * ( params_opt[1] * x_ri + 1
                            ) ** ( - params_opt[2])
    
    ## Plot data and best-fitting predictions
    _ = plt.figure( figsize=( 4.5, 4.5))
    plt.scatter( ri, rec, s=60, 
                 facecolors='none', edgecolors=Red)
    plt.plot( x_ri, pow_pred, color='k', linewidth=2)
    dev = pow_pred[ ri.astype('int')]
    for i in range( len(ri)):
        plt.plot( [ ri[i], ri[i]], [ dev[i], rec[i]],
                  color='k', linewidth=1)
    plt.xlabel( 'Retentional Interval (Days)')
    plt.ylabel( 'Proportion Items Retained')
    plt.xticks( np.arange( 0, np.max(ri)+2, 5))

    return params_opt


def bootstrapping( params_opt):
    '''
    Bootstapping algorithm:

        step1: generate a number of samples from the model
        step2: we then “simulate” as many “observed” proportions
    '''
    ## Get the human behavioral data from Carpenter et al 2008 exp1
    ri  = np.array([ .0035, 1, 2, 7, 14, 42])        # time interval 

    ## Define some variables
    ns = 55     
    nbs = 1000 
    bsparams = np.zeros( [ nbs, len( params_opt)])
    bspow_pred = params_opt[0] * ( params_opt[1] * ri + 1
                            ) ** ( - params_opt[2])
    
    ## Synthesize data
    for i in range(nbs):
        mean_fn = lambda x: np.round( np.random.binomial( ns, x) / ns, 2)
        recsynth = list(map( mean_fn, list(bspow_pred)))
        bsparams[ i, :] = fmin( powediscrep, 
                        params_opt, args=( recsynth, ri), disp=False)

    ## Visualize 
    x = np.linspace( 0, 1., 5)
    sz = 3.75
    param_name = [ 'a', 'b', 'c']
    _, axs = plt.subplots( 1, len(params_opt), figsize=( sz*len(params_opt), sz))
    for i in range( len(params_opt)):
        ax = axs[ i]
        params = bsparams[ :, i]
        ax.hist( params, bins=10, facecolor='none', edgecolor='k')
        ax.axvline( np.quantile( params, .025), 
                linestyle='dashed', color='k')
        ax.axvline( np.quantile( params, .975), 
                linestyle='dashed', color='k')
        ax.set_ylabel( 'Frequency')
        ax.set_xlabel( f'{param_name[i]}')
        ax.set_xlim([ -0.1, 1.1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

if __name__ == '__main__':

    ###########################
    ##       Example 1       ##
    ###########################
    
    ## generate fake data 
    data = synthesize_data()
    # show why the 
    plot_reg( data, fit_reg_params( data))
    plt.savefig( f'{path}/fig3-two snapshots for parameters.png', dpi=dpi)

    ###########################
    ##       Example 2       ##
    ###########################

    ## If we want to evaluate the variation of the
    #  parameter, but there is no data more data for
    #  us to feed more models. So we synthesize data.
    params_opt = forget_model()
    plt.savefig( f'{path}/fig1-forgeting model.png', dpi=dpi)
    bootstrapping( params_opt)
    plt.savefig( f'{path}/fig7-parameter variability.png', dpi=dpi)



    


