#Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Importing the dataset
dataset = pd.read_csv('USA_Housing.csv')
X = dataset.iloc[:,:-2].values
y = dataset.iloc[:,-2 ].values
#a = dataset.iloc[4999,:-2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict([X_test[1]])

# save the model to disk
filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


