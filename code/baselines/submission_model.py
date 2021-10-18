
# import packages

import numpy as np
import importlib
from sklearn.metrics import accuracy_score,\
 f1_score, precision_score, recall_score
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import proposed_model_keras_CNN_LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

importlib.reload(proposed_model_keras_CNN_LSTM)

from keras.models import load_model

class submission_model:
    
    def __init__(self):
        pass
    def train_model(self, x_train, y_train,\
                    x_validation, y_validation,\
                    x_test, y_test):

        y_train = to_categorical(y_train)
        y_validation = to_categorical(y_validation)
        
        
        # data shape
        input_shape = x_train.shape[1:]
        n_outputs = y_train.shape[1]

        # define model
        model = proposed_model_keras_CNN_LSTM.CNN_LSTM().\
        proposed_method(input_shape, n_outputs)

        # define optimizer parameter
        lr = 0.01
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-08
        decay = 0.001
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)

        # compile model
        model.compile(loss='categorical_crossentropy',\
                      optimizer=optimizer,\
                      metrics=['accuracy'])

        # define model parameter
        EPOCHS = 100
        BATCH_SIZE = 256 
        SHUFLE = True

        # checkpoints
        es = EarlyStopping(monitor='val_loss',\
                           mode='min',\
                           verbose=2,\
                           patience=25)
        mc = ModelCheckpoint('./submission_proposed_final.h5',\
                             monitor='val_acc',\
                             mode='max',\
                             verbose=2,\
                             save_best_only=True)
        # fit model
        _ = model.fit(x_train, y_train,\
                            epochs=EPOCHS, batch_size=BATCH_SIZE,\
                            validation_data=(x_validation, y_validation),\
                            shuffle=SHUFLE, callbacks=[es,mc]) 
        
        saved_model = load_model('./submission_proposed_final.h5')
        predictions = saved_model.predict(x_test)
        y_pred = predictions.argmax(axis=1)
        average = 'macro'
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        
        return accuracy, precision, recall, f1
        
        


