#This is based on:
#https://www.tensorflow.org/addons/api_docs/python/tfa/layers/esn and
#https://github.com/ashesh6810/RCESN_spatio_temporal/blob/master/ESN.py
#https://npg.copernicus.org/articles/27/373/2020/#bib1.bibx38

"""Implements Echo State recurrent Network (ESN) layer."""
import sys
import tensorflow as tf
#from tensorflow_addons.rnn import ESNCell

from tqdm import tqdm
import scipy
import numpy as np
from src.utils import inverse_transform
from src.ESNCell import ESNCell

#@tf.keras.utils.register_keras_serializable(package="Addons")
class ESN(tf.keras.layers.RNN):
    """Echo State Network layer.
    This implements the recurrent layer using the ESNCell.
    This is based on the paper
        H. Jaeger
        ["The "echo state" approach to analysing and training recurrent neural networks"]
        (https://www.researchgate.net/publication/215385037).
        GMD Report148, German National Research Center for Information Technology, 2001.
    Arguments:
        units: Positive integer, dimensionality of the reservoir.
        connectivity: Float between 0 and 1.
            Connection probability between two reservoir units.
            Default: 0.1.
        leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it's the special case the model does not have leaky integration.
            Default: 1.
        spectral_radius: Float between 0 and 1.
            Desired spectral radius of recurrent weight matrix.
            Default: 0.9.
        use_norm2: Boolean, whether to use the p-norm function (with p=2) as an upper
            bound of the spectral radius so that the echo state property is satisfied.
            It  avoids to compute the eigenvalues which has an exponential complexity.
            Default: False.
        use_bias: Boolean, whether the layer uses a bias vector.
            Default: True.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            Default: `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
            Default: `glorot_uniform`.
        bias_initializer: Initializer for the bias vector.
            Default: `zeros`.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    Call arguments:
        inputs: A 3D tensor.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the cell
            when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.
     """
    #@typechecked
    def __init__(
        self,

        input_layer: str,
        output_layer: str,

        units: int,

        exponents,
        coefficients,

        connectivity: float = 0.1,
        leaky: float = 1,
        spectral_radius: float = 0.9,
        use_norm2: bool = False,
        use_bias: bool = True,
        activation = tf.keras.activations.tanh, #: Activation = "tanh",
        kernel_initializer = tf.keras.initializers.glorot_uniform(), #: Initializer = "glorot_uniform",
        recurrent_initializer = tf.keras.initializers.glorot_uniform(), #: Initializer = "glorot_uniform",
        bias_initializer = tf.keras.initializers.Zeros(), #Initializer = "zeros",
        return_sequences=False,
        go_backwards=False,
        unroll=False,
        
        sparse: bool = False,
        sigma: float = 0.5, 

        **kwargs
    ):  
        self.input_layer = input_layer.split( ":", 1 )[ 0 ].split( "/", 1 )[ 0 ] #tf.keras.Input gives names like "input_layer:0"
        self.output_layer = output_layer
        self.exponents = exponents
        self.coefficients = coefficients

        cell = ESNCell(
            units,
            connectivity=connectivity,
            leaky=leaky,
            spectral_radius=spectral_radius,
            use_norm2=use_norm2,
            use_bias=use_bias,
            activation=activation,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dtype=kwargs.get("dtype"),

            sparse = sparse,
            sigma = sigma
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            go_backwards=go_backwards,
            unroll=unroll,
            **kwargs,
        )


    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
            constants=None,
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def connectivity(self):
        return self.cell.connectivity

    @property
    def leaky(self):
        return self.cell.leaky

    @property
    def spectral_radius(self):
        return self.cell.spectral_radius

    @property
    def use_norm2(self):
        return self.cell.use_norm2

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def activation(self):
        return self.cell.activation

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    def get_config(self):
        config = {
            "units": self.units,
            "connectivity": self.connectivity,
            "leaky": self.leaky,
            "spectral_radius": self.spectral_radius,
            "use_norm2": self.use_norm2,
            "use_bias": self.use_bias,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": tf.keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def reservoir_layer( self, A, Win, leaky, input, train_length ):
        states = np.zeros( ( self.units, train_length ) )
        for i in tqdm( range( train_length - 1 ), desc="Regression in {} Dimensions ".format( input.shape[ 0 ] ) ):
            states[:, i+1] = ( 1 - leaky ) * states[:, i] + leaky * ( np.tanh(A.dot(states[:, i])+np.dot(Win, input[:, i])) )
        return states

    def train_reservoir_sparse( self, A, Win, leaky, data ):
        states = self.reservoir_layer( A, Win, leaky, data, data.shape[ -1 ] ) #=train_length
        Wout = self.train( states, data )
        return Wout

    def train( self, states, data):
        beta = 1e-4 #Tikhonov regularization
        idenmat = beta * scipy.sparse.identity( self.units )
        
        '''states2 = states.copy()
        for j in range(0, np.shape(states2)[0]-0): 
            if (np.mod(j, 2) == 0):
                states2[j, :] = (states[j-0, :]*states[j-0, :]).copy()
        states2_corr = states2.copy()
        '''

        coeff = self.coefficients
        coeff = np.tile( np.transpose( coeff ), states.shape[-1] ) 
        coeff = np.reshape( coeff, states.shape, order = 'F' )
        
        exp = self.exponents
        exp = np.tile( np.transpose( exp ), states.shape[-1] ) 
        exp = np.reshape( exp, states.shape, order = 'F' )
                
        states2 = np.multiply( coeff, states.copy() ** exp ) 

        states2 = np.where( ~ np.isinf( states2 ), states2, 0  )

        #print( "\n\nstates2:\n\n\n", states2 )

        U = np.dot( states2, states2.transpose() ) + idenmat
        Uinv = np.linalg.inv( U )

        return np.dot( Uinv, np.dot( states2, np.transpose( data ) ) )

    def train_ESN_Output( self, data, model, esn_hyperparameter, sparse, time_length ): 

        kernel = tf.identity( model.get_layer( self.name ).cell.kernel )
        recurrent_kernel = tf.identity( model.get_layer( self.name ).cell.recurrent_kernel )

        if sparse:
            kernel = tf.sparse.to_dense( kernel )
            recurrent_kernel = tf.sparse.to_dense( recurrent_kernel )

        Win = np.transpose( kernel )
        #Win = scipy.sparse.csr_matrix( np.transpose( kernel) ) #doesn't work
        #A = np.transpose( recurrent_kernel ) #slow
        A = scipy.sparse.csr_matrix( np.transpose( recurrent_kernel ) )

        try:
            model.get_layer( name = self.input_layer )
        except:
            self.input_layer = "tf_op_layer_" + self.input_layer

        intermediate = tf.keras.Model( inputs = model.input, outputs = model.get_layer( name = self.input_layer ).output )

        inputs = np.array( intermediate.predict( data ) )
        
        inputs = inverse_transform( inputs )#, esn_hyperparameter["bs"] )

        Wout = self.train_reservoir_sparse( A, Win, esn_hyperparameter["leaky"], inputs[ :, 0 * time_length: ( 0 + 1 ) * time_length ] )
        #old = np.array( model.get_layer( self.output_layer ).get_weights() ).shape

        model.get_layer( self.output_layer ).set_weights( [ Wout ] )

        #new = np.array( model.get_layer( self.output_layer ).get_weights() ).shape

