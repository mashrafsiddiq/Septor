from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
        X_train, Y_train, test_size=test_size, random_state=seed)


# Create XGB Classifier object
model = XGBClassifier(verbosity=1,num_classes=6,
                      objective='multi:softmax')
# Fit model
model.fit(X_train, y_train,verbose=True)
# Predictions
y_train_preds = model.predict(X_train)
y_test_preds = model.predict(X_test)


print("Training F1 Micro Average: ",
      f1_score(y_train, y_train_preds, average = "micro"))
print("Test F1 Micro Average: ",
      f1_score(y_test, y_test_preds, average = "micro"))