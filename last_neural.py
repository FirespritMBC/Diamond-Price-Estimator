# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 00:27:29 2022

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

#Training Data
start_time = time()
nn = NeuralNetwork(neuron_number_in_hidden_layer = 16, num_of_features = 9, learning_rate = 0.001) 
loss_history = nn.training(train_X, train_Y)
print("Learning rate is: 0.001")
print("Neuron number in hidden layer is: 16")
print("It has taken {0} seconds to train the network".format(time() - start_time))

plt.figure()
plt.plot(loss_history)
plt.title("Epoch num vs training MSE") 
plt.xlabel("Epoch num") 
plt.ylabel("Training MSE")

test_Y_predicted = nn.predict(test_X) 
print("The Score: ", (R_squared(test_Y_predicted, test_Y)))

plt.figure()
plt.plot(test_Y, label="line1")
plt.plot(test_Y_predicted, label="line2")