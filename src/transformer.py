
from inspect import Attribute
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np 

class Transformer( layers.Layer ):        
    def __init__( self, exponents, coefficients ): #__init__
        super( Transformer, self ).__init__()
        
        self.exponents = self.add_weight( name = 'Exponents', shape = exponents.shape, dtype = self.dtype, trainable = False )
        self.coefficients = self.add_weight( name = 'Coefficients', shape = coefficients.shape, dtype = self.dtype, trainable = False )
        self.set_weights( list( [ exponents, coefficients ] ) ) #tf.math.multiply( self.exponents, exponents )

    def call( self, inputs ): 
        t = tf.math.multiply( self.coefficients, tf.math.pow( inputs, self.exponents ) )
        r =  tf.where( ~ tf.math.is_nan( t ), t, 0  )
        #tf.print( "\n\nreturn:\n\n\n", r )
        return r