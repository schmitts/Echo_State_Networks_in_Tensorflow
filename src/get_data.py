import numpy as np
from src.lorenz96 import calcLorenz
from src.utils import normalize
import pandas as pd 
import os

path = os.getenv("HOME") + '/ownCloud/Master_Arbeit/Mine/'

def Lorenz96( system_params ):
    return calcLorenz( min( 32, system_params["N"] ) , system_params["F"], tmax = system_params["tmax"], 
                             #dt = system_params["dt"], disP=[3], disStr=[0.01], offSetT=0)
                            dt = system_params["dt"], disP=[0], disStr=[0.01], offSetT=0)

def Lorenz96_new( system_params ):
    return np.transpose( pd.read_csv( path + '3tier_lorenz_v3.csv', header=None) )
                       
def KSE( system_params ):
    n_ = 70#15
    p = os.getenv("HOME") + '/ownCloud/Master_Arbeit/Mine/kuramotoSivashinsky/train/'    
    arr = np.zeros( ( system_params["N"], n_ * 5000 ) )
    for i in range( n_ ):
        package = np.load( p + 'KuramotoSivashinsky_L_200_Q_512_c_{:05.0f}.npy'.format( i + 1 )  
        )[ : system_params["N"], : 5000 ]
        arr[ :, i * 5000 : ( i + 1) * 5000 ] = package
    return arr

def KSE_new( system_params ):
    N = system_params["tmax"]
    dt = system_params["dt"]

    p = os.getenv("HOME") + '/ownCloud/Master_Arbeit/Mine/genKSE/'
    return np.load( p + 'KuramotoSivashinsky_L_200_Q_512_Nt_{}_dt_{}_c_00001.npy'.format( N, dt ) )

def get_data( system_params, train_params, ODE ):
    switcher = {
        0 : Lorenz96, 
        1 : Lorenz96_new,
        2 : KSE,
        3 : KSE_new
    }
    inputs = np.array( switcher.get( ODE )( system_params ) )
    n_samples = train_params["train_length"] + train_params["validation_length"]
    return inputs[ : system_params["N"], : n_samples ]
