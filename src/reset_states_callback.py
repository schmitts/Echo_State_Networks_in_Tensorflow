import tensorflow as tf
from src.utils import reset_states

class reset_states_callback( tf.keras.callbacks.Callback ):
    def __init__( self):
        super().__init__()
    def on_batch_begin(self, epoch, logs=None):
        reset_states( self.model )
        print( "cut" )
