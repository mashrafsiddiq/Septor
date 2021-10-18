from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Permute, Flatten
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import Reshape, Dense, Input, Dropout, Activation, LSTM, Conv2D,\
                            BatchNormalization, GRU, TimeDistributed, Bidirectional, Layer, Flatten
from tensorflow.keras import initializers
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np

import sys
    
class WAVEFORM():
    def __init__(self):
        pass
    
    def waveform_aggregator(self, input_shape):
        waveform_input = Input(shape=input_shape)
        
        X = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='same', name='conv1')(waveform_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)
        
        X = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', name='conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)
        
        X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', name='conv3')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)
        
        X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv4')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        
        X = Permute((2,1,3))(X)
        X = Permute((1,3,2))(X)
        shape = K.int_shape(X)   
        X = Reshape((shape[1], shape[2] * shape[3]))(X)
    
        X = LSTM(units=32, recurrent_dropout=0.3, return_sequences=True)(X)
        X = LSTM(units=32, recurrent_dropout=0.3, return_sequences=True)(X)

        X = Flatten()(X)
        X = Dense(2048)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dense(256)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dense(32)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        
        waveform_encode = Model(waveform_input, X)
        waveform_encode.summary()
        
        return waveform_encode
    
    
class STATION():
    def __init__(self):
        pass
    
    def station_aggregator(self, input_shape, dr):
        print(input_shape)
        wf_aggregator = WAVEFORM().waveform_aggregator((input_shape[1], input_shape[2], input_shape[3]))        
        station_input = Input(shape=input_shape, name='input_x')
        
        X = TimeDistributed(wf_aggregator)(station_input)
        shape = K.int_shape(X)
        X = Reshape((shape[1], shape[2], 1))(X)
        X = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', name='conv2')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(1,2), strides=(2,2), padding='same')(X)
        X = Permute((2,1,3))(X)
        X = Permute((1,3,2))(X)
        shape = K.int_shape(X)   
        X = Reshape((shape[1], shape[2] * shape[3]))(X)
        X = LSTM(units=32, recurrent_dropout=0.3, return_sequences=True)(X)
        X = Flatten()(X)
        X = Dense(64)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
        X = Dense(32)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
        X = Dense(1, kernel_initializer='glorot_normal')(X)
        
        station_encode = Model(station_input, X)
        
        return station_encode
