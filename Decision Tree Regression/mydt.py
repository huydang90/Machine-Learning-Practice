#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:52:40 2019

@author: dangngochuy
"""

#Import libraries 
import numpy as np 
import matplotlib.pyplot as plt #plot charts
import pandas as pd #best to import and manage datasets 

#Preprocess data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting Decision Tree  to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#Predict new results with Linear Regression 
y_pred = regressor.predict([[6.5]])

#Visualize the Decision Tree results 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y)
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Truth or Bluff Decision Tree")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()

