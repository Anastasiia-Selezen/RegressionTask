import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Load data sets
data = pd.read_csv('internship_train.csv')
test = pd.read_csv('internship_hidden_test.csv')

# Divide data into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Divide data set for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build and fit the model
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Evaluate model with 5-fold cross-validation to avoid overfitting
cv_results = cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

# Evaluate the model on train and test set
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# print Root mean squared error
print("RMSE for train data:"+str(rmse_train))
print("RMSE for test data:"+str(rmse_test))

# Generate and save predictions for internship_hidden_test.csv
predictions = regressor.predict(test.values)
np.savetxt("predictions.csv", predictions, delimiter=",")

