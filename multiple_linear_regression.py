import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

#convert column into categorical column
states = pd.get_dummies(X['State'], drop_first=True)
X = X.drop('State', axis = 1)
#concat dummies
X = pd.concat([X, states], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
score = r2_score(y_test, y_pred)