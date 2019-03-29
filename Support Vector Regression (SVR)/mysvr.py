#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:25:34 2019

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

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X= sc_X.fit_transform(X)
y = np.array(y).reshape(-1, 1)
y = sc_y.fit_transform(y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X, y)

#Predict new result: 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualize the SVR
plt.scatter(X,y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Truth or Bluff SVR")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()


#Visualize the SVR results (with higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y)
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Truth or Bluff SVR")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()
