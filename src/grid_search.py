# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys 
#sys.path.append('/home/jmeyer/git/jan_meyer_esn')
sys.path.insert(0,'..')

import glob
import datetime

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

import datetime
import itertools

from scipy.optimize import curve_fit

from src.get_data import get_data
from src.utils import *
from src.model import *
from src.train_ESN_callback import train_ESN_callback
from src.reset_states_callback import reset_states_callback

import matplotlib.pyplot as plt

#import tikzplotlib #https://pypi.org/project/tikzplotlib/

plt.rcParams["figure.figsize"] = (10,7)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams.update({'font.size': 14})

import seaborn as sns

import argparse

parser = argparse.ArgumentParser( description="Script to run Gridsearch" )
parser.add_argument('--min_exp', required=True,type=int )
parser.add_argument('--max_exp', required=True,type=int )
parser.add_argument('--min_coeff', required=True,type=int )
parser.add_argument('--max_coeff', required=True,type=int )
parser.add_argument('--periodicity', required=True,type=int )
args = parser.parse_args()



# %%
simple = not True


tf.random.set_seed(1234)    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# %%
if simple == True:
    esn_hyperparameter = { "nodes": 3000, "leaky": 1.0 - 1.000000082740371e-10, "kappa": 20, 
                        "spectral_radius": 5.0, "sigma": 0.3423630299505395, "bs":1, 
                        "n_esn": 2, "overlay_ratio": 0.025, "stateful": True }
    train_params = { "epochs": 1, #overlay=0.025 -> l = 6
                  "train_length": int( 10e3 ), "validation_length": int( 10e3 ),
                  "loss_function": 'mse', "sparse": True,
                  "optimizer": tf.keras.optimizers.Adam( 1e-6 ), "lr_decay": 5e-2, "output_by_gradient": False } 
else:
    esn_hyperparameter = { "nodes": 5000, "leaky": 1.0, "kappa": 3, 
                    "spectral_radius": 0.6, "sigma": 0.2, "bs":1, 
                    "n_esn": 1, "overlay_ratio": 0.025, "stateful": True } #n_esn must be true denominator of system_params["N"]
    train_params = { "epochs": 1, #overlay=0.025 -> l = 6
                  "train_length": int( 50e3 ), "validation_length": int( 10e3 ),
                  "loss_function": 'mse', "sparse": True,
                  "optimizer": tf.keras.optimizers.Adam( 1e-6 ), "lr_decay": 5e-2, "output_by_gradient": False } 

system_params = { "N": 64, #512, 
                  "F": 8, 
                  #"dt": 0.002, "tmax": 0.002 + 50000,
                  "dt": 0.25, "tmax": int( 1e5 ),
                  "time_length": train_params["train_length"] } #5000 for KSE from Sebastian!

n_samples = train_params["train_length"] + train_params["validation_length"]
validation_split = train_params["validation_length"] / n_samples
batch_size = int( system_params["time_length"] / 20 ) #1

warmup_steps = 100

how_much = min( int ( 100 / system_params["dt"] ), train_params["validation_length"] ) #approx 4 lyapunovtimes

train_params["validation_length"] = how_much 

prediction_steps = how_much

lambda_max = 0.09
print( how_much * system_params["dt"] * lambda_max )
n_lyap = int( how_much * system_params["dt"] * lambda_max)
print( n_lyap )


# %%
results = os.getenv("HOME") + '/git/jan_meyer_esn/Results/HP-Opt-Small_Gridsearch_transformer_{}.txt'.format( datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") )
print( results )

# %%
f = os.getenv( "HOME" ) + "/ownCloud/Master_Arbeit/Mine/genKSE_simple/*" #"/ownCloud/Master_Arbeit/Mine/genKSE/KuramotoSivashinsky_L_200_Q_512_Nt_*_dt_{}".format( system_params["dt"] ) 

timelines = glob.glob( f + "_l2*c_00001.npy" )

print( f, "\n" )
print( len(timelines ) )
print( timelines )


# %%
noise = []
for i in range( len( timelines ) ):
    noise.append( float( timelines[ i ].split( "l2_" )[ 1 ].split( "_c_" )[ 0 ] ) )
#sigmas
noise = np.array( noise )
print( len(noise) )
#noise


# %%
p = ( noise / np.amax( 100. * noise ) ).argsort()
#p


# %%
timelines = np.array( timelines )
timelines = timelines[ p ]
noise = noise[ p ]

len( timelines )

#noise


# %%
#for i in range( len(timelines) ):
#   print( timelines[i] )
print( timelines[ 0 ], timelines[ 1 ])


# %%
#inputs = get_data( system_params, train_params, 3 ) #1
n_samples = train_params["train_length"] + train_params["validation_length"]

inputs = np.load( timelines[ 0 ] )[ :, : n_samples ]
inputs.shape


# %%
#inputs_2 = np.load( timelines[ 3 ] )[ :, : n_samples ]
inputs_2 = np.load( 
    timelines[ -1 ]
    #[ 2 ] 
    )[ :, : train_params["train_length"] + how_much ]

inputs_2.shape


# %%
inputs = transform( normalize( inputs ) )
inputs.shape


# %%
inputs_2 = transform( normalize( inputs_2 ) )
inputs_2.shape


# %%
def scheduler( epoch, lr ):
    return  lr * tf.math.exp( -1* train_params["lr_decay"] ) 
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler( scheduler, verbose=0 ) 


# %%
model = esn_model( esn_hyperparameter, train_params, system_params )
#tf.keras.utils.plot_model( model, show_shapes = True )

#model2 = tf.keras.models.clone_model( model ) #does not work


# %%
'''
from sklearn.model_selection import GridSearchCV    #Needs manual scoring -> redundance
from sklearn.metrics import mean_squared_error
param_grid = dict(epochs=[10,20,30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit( inputs[ : -1 ], inputs[ 1 : ] ) 
'''


# %%
exp_test = np.ones( esn_hyperparameter["nodes"], dtype=np.int32 )
coeff_test = np.ones( esn_hyperparameter["nodes"], dtype=np.int32 )




# %%
periodicity = args.periodicity
def create_grid( min_exp = 1, max_exp = 2, min_coeff = 1, max_coeff = 1, periodicity = 2 ):
    exp = np.arange( min_exp, max_exp + 1 )
    print( "\nExponents: \n", exp )

    exp = np.tile( exp, ( periodicity, 1 ) )
    exp_all = list( itertools.product( *exp ) ) 
    print( "\nAll combinations for exponents with periodicity {}\n\n".format( periodicity ), exp_all )

    coeff = np.arange( min_coeff, max_coeff + 1 )
    print( "\nCoefficients: \n", coeff )

    coeff = np.tile( coeff, ( periodicity, 1 ) )
    coeff_all = list( itertools.product( *coeff ) ) 
    print( "\nAll combinations for coefficients with periodicity {}\n\n".format( periodicity ), coeff_all )

    combi = list( itertools.product( exp_all, coeff_all ) )
    print( "\nAll {} combinations of exponents and coefficients: \n\n".format( len( combi ) ), combi )
    return list( reversed( combi ) ) #s_

grid = create_grid( min_exp = args.min_exp, max_exp = args.max_exp, min_coeff = args.min_coeff, max_coeff = args.max_coeff, periodicity = args.periodicity )


# %%

nodes = esn_hyperparameter["nodes"]

#results = Parallel(n_jobs=2)(delayed(process)( grid[ i ] ) for i in range( len( grid ) ) )
for i in range( len( grid ) ):
    #process( grid[ i ] )
    
    ( exp, coeff ) = grid[ i ]
    exp = np.tile( exp, int( np.ceil( nodes / periodicity ) ) )[ : nodes ]
    coeff = np.tile( coeff, int( np.ceil( nodes / periodicity ) ) )[ : nodes ]

    print( "\nexp: \n", exp[ : 30 ] )
    print( "coeff: \n", coeff[ : 30 ] )
    alter_transformer( model, exp, coeff )

    #tf.random.set_seed(1234)
    #model = esn_model( esn_hyperparameter, train_params, system_params )
    model.fit( inputs[ : -1 ], inputs[ 1 : ], # watch_dim
                    batch_size = batch_size, epochs = train_params["epochs"], 
                    shuffle = False, validation_split = validation_split, verbose = 1,
                    callbacks = [ 
                                    train_ESN_callback( inputs[ : train_params["train_length"] ], esn_hyperparameter, train_params["sparse"], system_params["time_length"] ), 
                                ] 
                    )
    c = pred_horizon( model, esn_hyperparameter, system_params, inputs[ train_params["validation_length"] : ] )
    HP = esn_hyperparameter
    #HP['horizon'] = c
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")

    g = open( results, 'a' )

    g.write( '#date and time = ' + dt_string + ':\n' + str( HP ) + ',\n' )
    g.write( 'Exponents: \n' + str( exp[ : 30 ] ) )
    g.write( ' ... \nCoefficients: \n' + str( coeff[ : 30 ] ) + ' ... \n')
    g.write( '\nHorizon: {}\n\n'.format( c ) )

    g.close()
    
'''
from joblib import Parallel, delayed
def process( pair ):
    ( exp, coeff ) = grid[ i ]
    exp = np.tile( exp, int( np.ceil( nodes / periodicity ) ) )[ : nodes ]
    coeff = np.tile( coeff, int( np.ceil( nodes / periodicity ) ) )[ : nodes ]

    tf.random.set_seed(1234)
    model = esn_model( esn_hyperparameter, train_params, system_params )

    print( "\nexp: \n", exp[ : 30 ] )
    print( "coeff: \n", coeff[ : 30 ] )
    alter_transformer( model, exp, coeff )

    model.fit( inputs[ : -1 ], inputs[ 1 : ], # watch_dim
                    batch_size = batch_size, epochs = train_params["epochs"], 
                    shuffle = False, validation_split = validation_split, verbose = 1,
                    callbacks = [ 
                                    train_ESN_callback( inputs[ : train_params["train_length"] ], esn_hyperparameter, train_params["sparse"], system_params["time_length"] ), 
                                ] 
                    )
    c = pred_horizon( model, esn_hyperparameter, system_params, inputs[ train_params["validation_length"] : ] )
    HP = esn_hyperparameter
    #HP['horizon'] = c
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")

    #Individuell dann mergen
    g = open( results, 'a' )

    g.write( '#date and time = ' + dt_string + ':\n' + str( HP ) + ',\n' )
    g.write( 'Exponents: \n' + str( exp[ : 30 ] ) )
    g.write( ' ... \nCoefficients: \n' + str( coeff[ : 30 ] ) + ' ... \n')
    g.write( '\nHorizon: {}\n\n'.format( c ) )

    g.close()
'''