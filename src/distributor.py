import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Distributor( layers.Layer ):        
    def __init__( self, n_out : int, overlay : int ):
        super( Distributor, self ).__init__() 
        self.n_out = n_out
        self.overlay = overlay

    def build(self, inputs_shape):
        self.N = inputs_shape[ -1 ]

    def call( self, inputs ): 
                
        outputs = []
        for i in range( self.n_out ):
            feed = int( np.floor( self.N // self.n_out ) ) #Caution!
            x = tf.concat( [ inputs, inputs, inputs ], axis = -1 )
            outputs.append( x[ :, :, self.N + i * feed - self.overlay : self.N + ( i + 1 ) * feed + self.overlay ] ) #Periodic Boundaries missing

        outputs = tf.stack( outputs, 0 )
        #print( outputs.shape )
        return outputs