import tensorflow as tf
from tensorflow.keras import layers, backend as K
from src.ESNLayer import ESN
from src.transformer import Transformer
from src.distributor import Distributor
import numpy as np
from tqdm import tqdm

def exponents( nodes, dtype ):
    #return tf.ones( nodes, dtype = dtype ) 
    return tf.cast( ( tf.range( nodes ) + 1 ) % 2 + 1, dtype = dtype )

def coefficients( nodes, dtype ):
    return tf.ones( nodes, dtype = dtype )

def connector( esn_list, overlay ):
    esn_tensor = tf.convert_to_tensor( esn_list )
    if overlay > 0:
        esn_tensor = esn_tensor[ :, :, :, overlay : -overlay ]
    esn_tensor = list( tf.unstack( esn_tensor ) )
    combined = tf.keras.layers.concatenate( esn_tensor, axis=-1 )
    return combined

def esn_block( inputs, esn_hyperparameter, train_params ): #Can't use names multiple times

    outputs = layers.Dense( inputs.shape[ -1 ], activation = 'linear', use_bias = False,
                trainable = train_params["output_by_gradient"] )

    exp = exponents( esn_hyperparameter["nodes"], inputs.dtype )
    coeff = coefficients( esn_hyperparameter["nodes"], inputs.dtype )

    esn_layer = ESN( units = esn_hyperparameter["nodes"], leaky = esn_hyperparameter["leaky"], connectivity = esn_hyperparameter["kappa"] / esn_hyperparameter["nodes"], 
                use_bias = False, sparse = train_params["sparse"], sigma = esn_hyperparameter["sigma"], stateful = esn_hyperparameter["stateful"], 
                return_sequences = True, time_major = True, trainable = False, return_state = True, 
            input_layer = inputs.name, output_layer = outputs.name, exponents = exp, coefficients = coeff )

    T = Transformer( exponents = exp, coefficients = coeff )
    #T.exponents = np.array( exp )
    #T.coefficients = np.array( coeff )
    
    return outputs( T ( esn_layer( inputs )[ 0 ] ) )

def esn_parallel( input_data, esn_hyperparameter, train_params, system_params ):

    if esn_hyperparameter["n_esn"] == 1:
        return esn_block( input_data, esn_hyperparameter, train_params )
        
    overlay = tf.constant( int(  system_params[ "N" ] * ( 1.0 - 1.0 / esn_hyperparameter["n_esn"] ) * esn_hyperparameter["overlay_ratio"] / 2 ), dtype = tf.int32 )

    dist = Distributor( n_out = esn_hyperparameter["n_esn"], overlay = overlay )( input_data ) #pre )

    esn_list = []
    for i in tqdm( range( esn_hyperparameter["n_esn"] ), desc ="Finished ESN-Blocks" ): #tqdm(range( train_length - 1 ), desc="reservoir_layer")
        esn_list.append( esn_block( dist[ i ], esn_hyperparameter, train_params ) )

    return connector( esn_list, overlay )

def esn_model( esn_hyperparameter, train_params, system_params ):
    input_data = layers.Input( ( esn_hyperparameter["bs"], system_params[ "N" ] ) ) 
    
    #pre = layers.Dense( system_params[ "N" ], name = 'additional_dense_pre' )( input_data ) 

    combined = esn_parallel( input_data, esn_hyperparameter, train_params, system_params )

    #between = layers.Dense( system_params["N"], name = 'additional_dense_between' )( combined ) 

    #esn = esn_block( between, esn_hyperparameter, train_params )

    #post = layers.Dense( esn_hyperparameter["outputs"], name = 'additional_dense_post' )( esn_ ) 

    model = tf.keras.Model( inputs = input_data, outputs = combined ) 

    model.compile( loss = train_params["loss_function"], optimizer = train_params["optimizer"], metrics = [ 'mae' ] )              
    return model

def save_weights( model, sparse: bool, name: str ): #Very slow!

    if sparse:
        kernels = []
        recurrent_kernels = []
        for layer in reversed( model.layers ) : 
            if 'esn' in layer.name:
                l = model.get_layer( layer.name ).cell
                kernel = tf.sparse.to_dense( l.kernel )
                recurrent_kernel = tf.sparse.to_dense( l.recurrent_kernel )

                kernels.append( kernel ) #, tf.sparse.to_dense( l.recurrent_kernel )
                recurrent_kernels.append( recurrent_kernel ) #, tf.sparse.to_dense( l.recurrent_kernel )

        np.save( name + "_kernels.npy", kernels )
        np.save( name + "_recurrent_kernels.npy", recurrent_kernels )

    model.save_weights( name + "_dense_weights")

def load_weights( model, sparse: bool, name: str ): #Very slow!

    model.load_weights( name + "_dense_weights" )

    if sparse:
        kernels = np.load( name + "_kernels.npy" )
        recurrent_kernels = np.load( name + "_recurrent_kernels.npy" )

        i = 0

        for layer in reversed( model.layers ) : 
            if 'esn' in layer.name and sparse:
                cell = model.get_layer( layer.name ).cell
                cell.kernel = tf.sparse.from_dense( kernels[ i ] ) 
                cell.recurrent_kernel = tf.sparse.from_dense( recurrent_kernels[ i ] ) 
                i += 1




