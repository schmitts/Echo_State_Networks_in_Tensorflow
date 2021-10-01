import tensorflow as tf
from datetime import datetime
from src.utils import pred_horizon, transform, inverse_transform, rmse, horizon, future_loop, normalize, plot, plot_loss
from src.model import reset_states

import kerastuner as kt
import numpy as np

class save_hp_callback( tf.keras.callbacks.Callback ):
    def __init__( self, file, validation_data, hypermodel, tuner, transformer = False ): #batch_size
        super().__init__()
        self.file = file
        self.hypermodel = hypermodel
        self.tuner = tuner
        self.validation_data = validation_data[ : 1000 ] # = 22.5 Lyapunov-times
        self.transformer = transformer

    def on_test_begin( self, epoch, logs=None ): 
        self.hypermodel.horizon = pred_horizon( self.model, self.hypermodel.esn_hyperparameter, self.hypermodel.system_params, self.validation_data )

    def on_epoch_end( self, epoch, logs=None ):
        print(".......................................................................................................................................")

        HP = self.hypermodel.esn_hyperparameter
        HP['mse'] = list( logs.values() )[ 0 ]
        HP['mae'] = list( logs.values() )[ 1 ]
        HP['val_mse'] = list( logs.values() )[ 2 ]
        HP['val_mae'] = list( logs.values() )[ 3 ]

        HP['horizon'] = self.hypermodel.horizon

        print( HP )
        f = open( self.file, 'a')
        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
        print("#date and time =", dt_string)
        f.write( "#date and time = " + dt_string + ':\n' + str( HP ) + ',\n' )
        
        if( self.transformer == True ):
            f.write( 'Exponents: \n' )
            np.savetxt(f, self.hypermodel.exp.reshape(-1,100).tolist(),fmt='%d' )
            f.write( '\nCoefficients: \n' )
            np.savetxt(f, self.hypermodel.coeff.reshape(-1,100).tolist(), fmt='%2.3f' )
            f.write( '\n' )

        f.close()
        print(".......................................................................................................................................")

