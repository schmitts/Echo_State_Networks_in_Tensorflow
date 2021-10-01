import tensorflow as tf

class train_ESN_callback( tf.keras.callbacks.Callback ):
    def __init__( self, train_data, esn_hyperparameter, sparse, time_length, epochs = None ): #batch_size
        super().__init__()
        self.training_data = train_data
        self.hp = esn_hyperparameter
        self.sparse = sparse
        self.time_length = time_length
        self.epochs = epochs

    
    def f_element( self, x, X ):
        try:
            return x in X
        except:
            print('Epochs = None')
            return False
    #def on_epoch_begin(self, epoch, logs=None): 
    def on_epoch_begin(self, epoch, logs=None):
        if self.epochs == None and epoch == 0 or self.f_element( epoch, self.epochs ):
            for layer in reversed( self.model.layers ) : #all '*esn*' - layers are trained in the (reversed) order in which they appear in the model
                if 'esn' in layer.name:
                    l = self.model.get_layer( layer.name )
                    self.model.trainable = False    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    print( layer.name + ': ' + l.input_layer + " -> " + l.output_layer )

                    if self.hp["stateful"]:
                        l.reset_states()

                    l.train_ESN_Output( self.training_data, self.model, self.hp, self.sparse, self.time_length ) #[ : ( epoch + 1 ) * self.batch_size ] #only sparse!
