from keras.models import Model
from keras.layers import Dense, Flatten, Activation
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

class submission_cnn_model():
    """ reproduction of the paper
        Seismic Event and Phase Detection Using 
        Time-Frequency Representation and 
        Convolutional Neural Networks
    """
    def __init__(self):
        pass

    
    def train_cnn(self, input_shape, n_outputs):

    		inputs = Input(shape=input_shape)
    
    		X = Conv2D(filters=16, kernel_size=(3,5), 
    			strides=(1,1), padding='same', name='conv1')(inputs)
    		X = Activation('relu')(X)
    		X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)


    		X = Conv2D(filters=16, kernel_size=(3,5),
    					strides=(1,1), padding='same', name='conv2')(X)
    		X = Activation('relu')(X)
    		X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)


    		X = Conv2D(filters=16, kernel_size=(3,5),
    					strides=(1,1), padding='same', name='conv3')(X)
    		X = Activation('relu')(X)
    		X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)


    		X = Conv2D(filters=16, kernel_size=(3,5),
    					strides=(1,1), padding='same', name='conv4')(X)
    		X = Activation('relu')(X)
    		X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    		X = Flatten()(X) 

    		X = Dense(192, activation='relu')(X)
    		outputs = Dense(n_outputs, activation='softmax')(X)
    
    		model = Model(inputs=inputs, outputs=outputs, name='submission_CNN')

    		return model
        