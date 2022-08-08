# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:05:48 2022

@author: mehme
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#import seaborn as sns

diamond_data = pd.read_csv("diamonds.csv")
#print("There are {0} entries in the dataset before preprocessing".format(diamond_data.shape[0]))
#diamond_data = diamond_data.dropna()
#print("There are {0} entries in the dataset after dropping rows with NaN values".format(diamond_data.shape[0]))

features = diamond_data.columns
#print("Features are: ")
#for i in range(0,len(features)):
#    print("{0}: {1}".format(i, features[i]))

plt.figure()
plt.title("Carat")
plt.hist(diamond_data.carat)

plt.figure()
plt.title("Cut")
plt.hist(diamond_data.cut)

plt.figure()
plt.title("Color")
plt.hist(diamond_data.color)

plt.figure()
plt.title("Clarity")
plt.hist(diamond_data.clarity)

plt.figure()
plt.title("Depth")
plt.hist(diamond_data.depth)

plt.figure()
plt.title("Table")
plt.hist(diamond_data.table)

plt.figure()
plt.title("Length x")
plt.hist(diamond_data.x)

plt.figure()
plt.title("Length y")
plt.hist(diamond_data.y)

plt.figure()
plt.title("Length z")
plt.hist(diamond_data.z)

plt.figure()
plt.title("Price")
plt.hist(diamond_data.price)

for i in range(53940):
    if diamond_data.cut[i] == "Ideal":
        diamond_data.cut[i] = 1
    elif diamond_data.cut[i] == "Premium":
        diamond_data.cut[i] = 0.75
    elif diamond_data.cut[i] == "Very Good":
        diamond_data.cut[i] = 0.5
    elif diamond_data.cut[i] == "Good":
        diamond_data.cut[i] = 0.25
    else:
        diamond_data.cut[i] = 0

for i in range(53940):
    if diamond_data.color[i] == "D":
        diamond_data.color[i] = 1
    elif diamond_data.color[i] == "E":
        diamond_data.color[i] = 0.83
    elif diamond_data.color[i] == "F":
        diamond_data.color[i] = 0.67
    elif diamond_data.color[i] == "G":
        diamond_data.color[i] = 0.5
    elif diamond_data.color[i] == "H":
        diamond_data.color[i] = 0.33
    elif diamond_data.color[i] == "I":
        diamond_data.color[i] = 0.16
    else:
        diamond_data.color[i] = 0   

for i in range(53940):
    if diamond_data.clarity[i] == "IF":
        diamond_data.clarity[i] = 1
    elif diamond_data.clarity[i] == "VVS1":
        diamond_data.clarity[i] = 0.86
    elif diamond_data.clarity[i] == "VVS2":
        diamond_data.clarity[i] = 0.72
    elif diamond_data.clarity[i] == "VS1":
        diamond_data.clarity[i] = 0.58
    elif diamond_data.clarity[i] == "VS2":
        diamond_data.clarity[i] = 0.44
    elif diamond_data.clarity[i] == "SI1":
        diamond_data.clarity[i] = 0.28
    elif diamond_data.clarity[i] == "SI2":
        diamond_data.clarity[i] = 0.14
    else:
        diamond_data.clarity[i] = 0   

#Finding parameters for Normalization
max_carat = max(diamond_data.carat)
min_carat = min(diamond_data.carat)
max_depth = max(diamond_data.depth)
min_depth = min(diamond_data.depth)
max_table = max(diamond_data.table)
min_table = min(diamond_data.table)
max_x = max(diamond_data.x)
min_x = min(diamond_data.x)
max_y = max(diamond_data.y)
min_y = min(diamond_data.y)
max_z = max(diamond_data.z)
min_z = min(diamond_data.z)
max_price = max(diamond_data.price)
min_price = min(diamond_data.price)

#Normalization
diamond_data.carat = (np.array(diamond_data.carat)-min_carat)/(max_carat-min_carat)
diamond_data.depth = (np.array(diamond_data.depth)-min_depth)/(max_depth-min_depth)
diamond_data.table = (np.array(diamond_data.table)-min_table)/(max_table-min_table)
diamond_data.x = (np.array(diamond_data.x)-min_x)/(max_x-min_x)
diamond_data.y = (np.array(diamond_data.y)-min_y)/(max_y-min_y)
diamond_data.z = (np.array(diamond_data.z)-min_z)/(max_z-min_z)
diamond_data.price = (np.array(diamond_data.price)-min_price)/(max_price-min_price)

shuffled = diamond_data.sample(frac = 1)
#shuffled.to_csv("preprocessed_diamond_data.csv", index=False)



#correlation = diamond_data.drop(['price'], axis=1).corr(method="pearson")
#plt.figure()
#sns.heatmap(correlation, annot=True)
#plt.title("pearson correlation")