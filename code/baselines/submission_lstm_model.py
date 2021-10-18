from keras.initializers import Orthogonal
from keras import backend as K
from keras.models import Model
from keras.layers import Permute ,Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Reshape, Dense, Input,\
    Dropout, Activation, LSTM, Conv2D, BatchNormalization 
    
class submission_lstm_model():
    
    def __init__(self):
        super().__init__()
        
    def train_lstm(self, input_shape, n_outputs):

        inputs = Input(shape=input_shape)
        
        
        X = Permute((2,1,3))(inputs)
        X = Permute((1,3,2))(X)
        shape = K.int_shape(X)   
        X = Reshape((shape[1], shape[2]*shape[3]))(X)
    
        X = LSTM(units=32, recurrent_dropout=0.3, return_sequences=True)(X)
        X = LSTM(units=32, recurrent_dropout=0.3, return_sequences=True)(X)

        X = Flatten()(X) 
        X = Dropout(0.2)(X)
        X = Dense(32)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(16)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(0.2)(X)
        output_layer_initializer = Orthogonal()
        outputs = Dense(n_outputs,
                        activation='softmax',
                        kernel_initializer=output_layer_initializer,
                        trainable=False)(X)

        
        
        model = Model(inputs=inputs, outputs=outputs, name='proposed')
        print('defined')
        return model