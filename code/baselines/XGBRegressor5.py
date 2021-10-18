from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import get_stats

from sklearn.metrics import mean_squared_error
import numpy as np

X_train = np.load('all_data_3_channel.npy')
Y_train = np.load('all_label_3_channel.npy')
print(X_train.shape)

seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
        X_train, Y_train, test_size=test_size, random_state=seed)


# Create XGB Classifier object
model = XGBRegressor()
# Fit model
model.fit(X_train, y_train,verbose=True)
# Predictions
y_train_preds = model.predict(X_train)
y_test_preds = model.predict(X_test)


# print statistics
correlation, rmse, p_val = get_stats(y_test, y_test_preds)

print(f"Pearson's Correlation coefficient: {correlation}, RMSE: {rmse}, p value: {p_val}")