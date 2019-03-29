#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:24:53 2019

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

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualize the linear Regression results 
plt.scatter(X,y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff Linear Regression")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()


#Visualize the Polynomial Regression results 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y)
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or Bluff Polynomial Regression")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()

#Predict new results with Linear Regression 
lin_reg.predict([[6.5]])

#Predict new results with Polynomial Regression 
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


