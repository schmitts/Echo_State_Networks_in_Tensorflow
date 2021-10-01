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

from src.get_data import get_data
from src.utils import transform, inverse_transform, rmse, horizon, future_loop, normalize, plot, plot_loss
#from src.model import esn_model, save_weights, load_weights, reset_states
from src.esn_hyper_model import esn_hyper_model
from src.train_ESN_callback import train_ESN_callback
from src.save_hp_callback import save_hp_callback
from src.hp_cleanup_callback import *
from src.reset_states_callback import reset_states_callback

from tensorflow.keras import layers, backend as K
from src.ESNLayer import ESN
from src.transformer import Transformer
from src.distributor import Distributor


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,7)

import kerastuner as kt
from kerastuner import HyperModel

import argparse

parser = argparse.ArgumentParser( description="Script to run BaysianOptimizer" )
parser.add_argument('--only_transformer', '-ot', help='This is a boolean flag.', type=eval, choices=[True, False], default='False' ) 
parser.add_argument('--simple_test', '-st', help='This is a boolean flag.', type=eval, choices=[True, False], default='False' )
parser.add_argument('--regression', '-reg', help='This is a boolean flag.', type=eval, choices=[True, False], default='True' ) 
parser.add_argument('--optimizer', '-opt', required=True, type=str, default='Bayes' )

parser.add_argument('--max_trials', '-t', required=True, type=int )
parser.add_argument('--num_initial_points', '-nip', required=True,type=int )
parser.add_argument('--beta', '-b', required=True, type=float )
parser.add_argument('--default_nodes', '-n', required=True, type=int )
parser.add_argument('--default_leaky', '-l', required=True, type=float )
parser.add_argument('--default_kappa', '-k', required=True, type=int )
parser.add_argument('--default_radius', '-r', required=True, type=float )
parser.add_argument('--default_sigma', '-s', required=True, type=float )
parser.add_argument('--default_log_2_n_esn', '-log', required=True, type=int)
parser.add_argument('--default_overlay_ratio', '-o', required=True, type=float )

args = parser.parse_args()

print( "\n\toptimizing hyperparameters: {} \n".format( not args.simple_test ) )
print( "\n\tRegression: {} \n".format( args.regression ) )

'''
esn_hyperparameter = { "nodes": 600, "leaky": 0.99999999999, "kappa": 3, 
                    "spectral_radius": 0.6, "sigma": 0.2, "bs":1, 
                    "n_esn": 1, "overlay_ratio": 0.025, "stateful": True } #n_esn must true denominator of system_params["N"]
'''

esn_hyperparameter = { "nodes": int( args.default_nodes ), "leaky": float( args.default_leaky ), "kappa": int( args.default_kappa ), 
                    "spectral_radius": float( args.default_radius ), "sigma": float( args.default_sigma ), "bs":1, 
                    "n_esn": int( 2**int( args.default_log_2_n_esn ) ), "overlay_ratio": float( args.default_overlay_ratio ), "stateful": True } #n_esn must true denominator of system_params["N"]

train_params = { "epochs": 1, #overlay=0.025 -> l = 6
                  "train_length": int( 50e3 ), "validation_length": int( 10e3 ),
                  "loss_function": 'mse', "sparse": True,
                  "optimizer": tf.keras.optimizers.Adam( 1e-6 ), "lr_decay": 5e-2, "output_by_gradient": False } 
system_params = { "N": 64 #512
                , "F": 8, 
                  #"dt": 0.002, "tmax": 0.002 + 50000,
                  "dt": 0.25, "tmax": int( 1e5 ),
                  "time_length": train_params["train_length"] } #5000 for KSE from Sebastion!

n_samples = train_params["train_length"] + train_params["validation_length"]
validation_split = train_params["validation_length"] / n_samples

batch_size = int( system_params["time_length"] / 20 ) #1

#f = os.getenv( "HOME" ) + "/ownCloud/Master_Arbeit/Mine/genKSE/KuramotoSivashinsky_L_200_Q_512_Nt_*_dt_{}".format( 0.25 ) 
f = os.getenv( "HOME" ) + "/ownCloud/Master_Arbeit/Mine/genKSE_simple/*" #"/ownCloud/Master_Arbeit/Mine/genKSE/KuramotoSivashinsky_L_200_Q_512_Nt_*_dt_{}".format( system_params["dt"] ) 

timelines = glob.glob( f + "_l2*c_00001.npy" )
timelines.sort()

print( len( timelines ) )

inputs = np.load( timelines[-1] )[ :, : n_samples ]
inputs = transform( normalize( inputs ) )
train = inputs[ : train_params["train_length"] ]
val = inputs[ train_params["train_length"] : ]

print( inputs.shape )
print( val.shape )

hypermodel = esn_hyper_model( esn_hyperparameter, train_params, system_params, args.simple_test, args.only_transformer )

max_trials = args.max_trials
num_initial_points = args.num_initial_points

if( args.only_transformer ):
    results = os.getenv("HOME") + '/git/jan_meyer_esn/Results/HP-Opt-Small_Transformer.txt'
else:
    results = os.getenv("HOME") + '/git/jan_meyer_esn/Results/HP-Opt-Results.txt'

now = datetime.datetime.now()
project_name = now.strftime( "%Y_%m_%d_%H_%M_%S_%f" )
print( project_name )

print( "tuner: ", args.optimizer )
if( args.optimizer == 'Bayes' ):
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective=kt.Objective('val_mae', 'min'),
        max_trials = max_trials,
        num_initial_points = num_initial_points,
        overwrite = True,
        directory = os.getenv( "HOME" ) + '/git/jan_meyer_esn/HP_Opt_logs',
        project_name = project_name,
        #beta = args.beta #Check kt Version
        )

if( args.optimizer == 'Random' ):
    tuner = kt.RandomSearch(
        hypermodel,
        objective=kt.Objective('val_mae', 'min'),
        max_trials = max_trials,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        directory = os.getenv( "HOME" ) + '/git/jan_meyer_esn/HP_Opt_logs',
        project_name = project_name,
        )
        
def callbacks():
    cb = []
    if( args.regression == True ):
        cb.append( train_ESN_callback( train, hypermodel.esn_hyperparameter, True, train.shape[0] ) )
	#cb.append( learning_rate_scheduler),
    #cb.append( tensorboard_callback ),
    #cb.append( reset_states_callback() ),
    if( args.only_transformer == True ):
        cb.append( save_hp_callback( results, val, hypermodel, tuner, transformer = True ) )
    else:
        cb.append( save_hp_callback( results, val, hypermodel, tuner ) )
    cb.append( hp_cleanup_callback( project_name ) )

    return cb

tuner.search( inputs[ : -1 ], inputs[ 1: ],
             batch_size = batch_size, epochs = 1,
             shuffle = False, validation_split = validation_split, verbose = 1,
             callbacks = callbacks()
             )

tuner.results_summary()

dir = os.getenv( "HOME" ) + '/git/jan_meyer_esn/HP_Opt_logs'  
path = os.path.join( dir, project_name ) 
#print( path )
findNremove( path, "", 1000000 ) #Empty Pattern
