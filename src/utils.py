import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import tensorflow as tf

def transform( data ):
    return np.expand_dims( data.transpose(), 1 )
    
def inverse_transform( data ):
    return data[ :, 0, : ].transpose()

def normalize( x ):
    x_ = x - np.mean( x ) 
    return np.array( x_ / np.std( x_ ) )

def rmse( arr ):
    return np.sqrt( np.mean( arr **2 ) )

def horizon( arr, sigma ):
    for i in range( arr.shape[-1] ):
        if rmse( arr[ :, i ] ) > sigma:
            return i
    return -1

def horizon_reduced( arr, sigma ):
    for i in range( arr.shape[-1] ):
        if rmse( arr[ i ] ) > sigma:
            return i
    return -1

def plot_loss( history ):
    plt.semilogy( history.history[ 'loss' ][ : ] )
    plt.semilogy( history.history[ 'val_loss' ][ : ] )
    plt.semilogy( history.history[ 'mae' ][ : ] )
    plt.semilogy( history.history[ 'val_mae' ][ : ] )

    #plt.semilogy( history.history[ 'lr' ] )
    plt.title( 'Model Loss' )
    plt.ylabel( 'Loss' )
    plt.xlabel( 'Epoch' )
    plt.legend( [ 'mse', 'val_mse', 'mae', 'val_mae' ], loc = 'lower left' )
    plt.show()
    
def plot( truth, pred, shift, length, n_lyap, N, L, Q, name=None ):
    fig, axs = plt.subplots( int( np.ceil( N/ 4 ) ), 4, figsize=(12, 6), sharex=True )

    x = np.arange( length ) * n_lyap / length
    ncol = 3

    for idx, ax in enumerate(axs.reshape(-1)):
        space = idx * max( 1, int( truth.shape[ 0 ] / N ) )
        groundtruth = truth[ space, shift : shift + length ]
        predD = pred[ space, shift : shift + length ]
        ax.plot( x, groundtruth, 'royalblue' )
        ax.plot( x, predD, 'maroon' )
        ax.set_ylabel( "$y({}, t )$".format( round(space * L / Q) ) )
        if idx >= N/2:
            ax.set_xlabel( '$\Lambda_{max}~t$' )

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xticks( np.arange( 0, n_lyap + 1, 2 ) )
        ax.grid( True, alpha = 0.5 )

        c = horizon_reduced( truth[ space ] - pred [ space ], 0.5 )

        if 0 < c < length:
            ax.axvline(x = c * n_lyap / length, c="black")
    
    plt.tight_layout( pad=0.5 )
    fig.legend(["Ground Truth", "Prediction", "Horizon"], ncol=ncol, loc = 'lower center', bbox_to_anchor=[0.5, -0.09], bbox_transform=fig.transFigure ) #, loc = 'lower right' ) #-0.2 #, bbox_to_anchor=( 0.5, 0.5)
    #fig.text(0.5, 0.025, r"$\Lambda_{max}~t$", ha='center', rotation='horizontal', fontsize=plt.rcParams['axes.labelsize']) #, va='center'

    if name != None:
        plt.savefig('./../Results/Prediction_{}.png'.format(name), bbox_inches='tight' )
        plt.savefig('./../Results/Prediction_{}.svg'.format(name), bbox_inches='tight' )
        #tikzplotlib.save( './../Results/Prediction_{}.tex'.format(name) )

    plt.show()
    
    print( c )
    print( c * n_lyap / length )
    

def future_loop( model, nmr_of_predictions, esn_hyperparameter, system_params, last_output, states = None ):
    #activity = np.zeros( ( nmr_of_predictions, 1, esn_hyperparameter["nodes"] ), 'float32' )
    pred = np.expand_dims( last_output, 0 )
    future = np.zeros( ( nmr_of_predictions, esn_hyperparameter["bs"], system_params[ "N" ] ), 'float32' )

    for i in range( nmr_of_predictions ): 
    
        #activity[ i ] = np.array( model.get_layer( 'esn' ).states[0][-1], 'float32' )

        pred = model.predict_on_batch( pred )
        future[ i ] = pred 

    return np.array( future )

def alter_transformer( model, exponents, coefficients ):
    for layer in reversed( model.layers ): #all '*esn*' - layers are trained in the (reversed) order in which they appear in the model
        if 'esn' in layer.name:
            l = model.get_layer( layer.name )
            l.exponents = np.array( exponents.copy(), dtype = l.dtype )
            l.coefficients = np.array( coefficients.copy(), dtype = l.dtype )
            print( "altered ", layer.name )

        if 'transformer' in layer.name:
            model.get_layer( layer.name ).set_weights( [ exponents, coefficients ] )
            print( "altered ", layer.name )

def pred_horizon( model, esn_hyperparameter, system_params, validation_data ):
    warmup_steps = 50
    pred_steps = 900

    reset_states( model )

    warmup = model.predict_on_batch( validation_data[ : warmup_steps ] )

    pred_0 = warmup[ -1 ]

    prediction = future_loop( model, pred_steps, esn_hyperparameter, system_params, pred_0 )
    pred = inverse_transform( prediction )
    val = inverse_transform( validation_data[ warmup_steps : warmup_steps + pred_steps ] )

    c = horizon( val - pred, 0.5 ) * 0.25 * 0.09 
    print( '\n\n\t Horizon: {}\n'.format( c ) )

    return c

def reset_states( model ):
    for layer in reversed( model.layers ): 
        if 'esn' in layer.name:
            model.get_layer( layer.name ).reset_states()