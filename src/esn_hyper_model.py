import sys

from tensorflow.python.autograph.pyct import transformer 
#sys.path.append('/home/jmeyer/git/jan_meyer_esn')
sys.path.insert(0,'..')

import glob
import datetime

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.get_data import get_data
from src.utils import transform, inverse_transform, rmse, horizon, future_loop, normalize, plot, plot_loss
from src.model import esn_model, save_weights, load_weights, reset_states
from src.train_ESN_callback import train_ESN_callback
from src.reset_states_callback import reset_states_callback

from tensorflow.keras import layers, backend as K
from src.ESNLayer import ESN
from src.transformer import Transformer
from src.distributor import Distributor

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,7)

import kerastuner as kt
from kerastuner import HyperModel

from tensorboard.plugins.hparams import api as hp #?
#https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams?hl=en #Does not work with keras tuner!!!

def exponents( nodes, dtype ):
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
    
    T = Transformer()
    T.exponents = np.array( exp )
    T.coefficients = np.array( coeff )

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

#https://keras-team.github.io/keras-tuner/#you-can-use-a-hypermodel-subclass-instead-of-a-model-building-function
class esn_hyper_model( HyperModel ):
    def __init__( self, esn_hyperparameter, train_params, system_params, simple_test, only_transformer ):
        self.esn_hyperparameter = esn_hyperparameter.copy()
        self.defaults = esn_hyperparameter.copy()
        self.train_params = train_params
        self.system_params = system_params
        self.simple_test = simple_test
        self.only_transformer = only_transformer

    def build( self, hp ):
        self.exp = np.empty( self.esn_hyperparameter["nodes"] )
        self.coeff = np.empty( self.esn_hyperparameter["nodes"] )

        if( self.only_transformer ):
            print("only_transformer \n", self.only_transformer )
            for i in range( self.esn_hyperparameter["nodes"] ): 
                self.exp[ i ] = hp.Int( 'exp_{:5.0f}'.format( i ), -2, 5, default = ( i + 1 ) % 2 + 1 )
            print( 'True exponents: \n', self.exp )

            for i in range( self.esn_hyperparameter["nodes"] ): 
                self.coeff[ i ] = hp.Float( 'coeff_{:5.0f}'.format( i ), -10, 10, default = 1 )
            print( 'True coefficients: \n', self.coeff )

        else:
            self.esn_hyperparameter["spectral_radius"] = hp.Float( "spectral_radius", 0.1, 5.0, default = self.defaults["spectral_radius"], sampling = 'log' )
            print( "simple_test: {}".format( self.simple_test ) )

            if( self.simple_test == False ):
                log_esn = int( np.log2( int( self.defaults["n_esn"] ) ) )
                print( "log_esn: {}".format( log_esn ) )
                
                self.esn_hyperparameter["nodes"] = hp.Int( "nodes", 500, 7000, default = self.defaults["nodes"] )
                self.esn_hyperparameter["kappa"] = hp.Int( "kappa", 2, 30, default = self.defaults["kappa"], sampling = 'log' )
                self.esn_hyperparameter["leaky"] = 1 - hp.Float( "1 - leaky", 1e-10, 5e-1, default = self.defaults["leaky"], sampling = 'log' )
                self.esn_hyperparameter["sigma"] = hp.Float( "sigma", 0.05, 1.0, default = self.defaults["sigma"] )
                self.esn_hyperparameter["n_esn"] = int( np.power( 2, hp.Int( "log2(n_esn)", 0, 6, default = log_esn ) ) ) 
                self.esn_hyperparameter["overlay_ratio"] = hp.Float( "overlay_ratio", 0.0, 0.08, default = self.esn_hyperparameter["overlay_ratio"] )

        print( "\nTrial Hyperparamters:\n", self.esn_hyperparameter, "\n" )

        input_data = layers.Input( ( self.esn_hyperparameter["bs"], self.system_params[ "N" ] ) ) 
        combined = esn_parallel( input_data, self.esn_hyperparameter, self.train_params, self.system_params )
        model = tf.keras.Model( inputs = input_data, outputs = combined )
        model.compile( loss = self.train_params["loss_function"], optimizer = self.train_params["optimizer"], metrics = [ 'mae' ] ) 

        if( self.only_transformer ):
            for layer in reversed( model.layers ) : #all '*esn*' - layers are trained in the (reversed) order in which they appear in the model
                    if 'esn' in layer.name or 'transformer' in layer.name:
                        l = model.get_layer( layer.name )
                        print( "set exp and coeff for layer name: ", layer.name )
                        l.exp = self.exp
                        l.coeff = self.coeff
        return model


