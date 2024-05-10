#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:43:32 2024

@author: cornelius
"""



# ---------------------------- IMPORTS ---------------------------------


from NeuralNet import NeuralNetwork

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("mushroom.csv")
pd.set_option('display.max_columns', None)

features = df.drop(columns="class")
labels   = df['class']

feature_train, feature_test, label_train, label_test = train_test_split(
    features,labels, test_size=0.99, random_state=10)


# ------------------------- NEURAL NETWORK -----------------------------

input_nodes = len(features.columns)
hidden_nodes = 200
output_nodes = len(set(labels))
learning_rate = 1e-2
epochs = 10


n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes,learning_rate)
n.fit(feature_train, label_train, epochs)
prediction = n.predict(feature_test)

accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction)
sensitivity = metrics.recall_score(label_test, prediction)
f1_score = metrics.f1_score(label_test, prediction)



print("The accuracy is: ", accuracy)
print("The precision is: ", precision)
print("The sensitivity is: ", sensitivity)
print("The F1 Score is: ", f1_score)






























