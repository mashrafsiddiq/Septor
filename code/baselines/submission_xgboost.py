from sklearn.metrics import accuracy_score,\
 f1_score, precision_score, recall_score
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
class submission_xgboost:
    
    def __init__(self):
        pass
    
    def train_model(self, x_train, y_train,\
                  x_validation, y_validation,\
                  x_test, y_test,\
                  average='macro'):
        
        x_train_len = len(y_train)
        x_train = np.reshape(x_train, (x_train_len,-1))
        
        x_validation_len = len(y_validation)
        x_validation = np.reshape(x_validation, (x_validation_len,-1))
        
        x_test_len = len(y_test)
        x_test = np.reshape(x_test, (x_test_len,-1))
        
        # normalize each column (each feature vector)
        scaler=MinMaxScaler()
        # fit a normalizer using training data and
        # then perform normalization on training data
        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)
        x_test = scaler.transform(x_test)
        # normalize validation data
        model = XGBClassifier(verbosity=1,num_classes=6,
                      objective='multi:softmax')
        # Fit model
        model.fit(x_train, y_train,verbose=True)
        # Predictions
        # perform prediction on the validation data
        y_pred = model.predict(x_test)
        # calculate accuracy on validation prediction
        # print validation accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        
        return accuracy, precision, recall, f1