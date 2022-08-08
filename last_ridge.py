# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 23:26:14 2022

@author: mehme
"""

#Importing Libraries and Functions defined
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from last_functions import *

#Getting the data
diamond_dataset = pd.read_csv("preprocessed_diamond_data.csv")
Y = np.array(diamond_dataset.price)                     #labels
X = np.array(diamond_dataset.drop("price",axis = 1))    #features

#Separating the dataset
length = Y.shape[0]
datum = np.arange(0, length)
test_length = int(np.round(0.25 * length))
print("Test size: {0}".format(test_length))    
validation_length = int(np.round(0.1 * length))
print("Validation size: {0}".format(validation_length))    
train_length = length - test_length - validation_length
print("Train size: {0}".format(train_length))

test_datum = datum[0: test_length]
validation_datum = datum[test_length: test_length + validation_length]
train_datum = datum[test_length + validation_length:]

test_Y = Y[test_datum]
test_X = X[test_datum, :]
validation_Y = Y[validation_datum]
validation_X = X[validation_datum, :]
train_Y = Y[train_datum]
train_X = X[train_datum, :]

#Training Data Direct
start1 = time()
training_mse, beta_values, bias_value = ridge_regression_algorithm_direct(train_X, train_Y, 0.05)
print("The training takes {0} seconds".format(time() - start1))
test_Y_predicted = bias_value + test_X.dot(beta_values)
print("The Score: ", (R_squared(test_Y_predicted, test_Y)))

plt.figure()
plt.plot(test_Y, label="line1")
plt.plot(test_Y_predicted, label="line2")

#Training Data with Gradient Descent
start2 = time()
MSE_values_array, beta_values, bias_value = ridge_regression_algorithm(train_Y, train_X, 0.2, 0.1, 5000)
print()
print("The training takes {0} seconds".format(time() - start2))
print("MSE is: {0}".format(MSE_values_array[-1]))

plt.figure()
plt.plot(MSE_values_array,'r')
plt.title("MSE Values of Train Set")
plt.xlabel("Iteration")
plt.ylabel("MSE Values")

test_Y_predicted = bias_value + test_X.dot(beta_values)
print("The Score: ", (R_squared(test_Y_predicted, test_Y)))

plt.figure()
plt.plot(test_Y, label="line1")
plt.plot(test_Y_predicted, label="line2")