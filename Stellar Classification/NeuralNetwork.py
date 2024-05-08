#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:22:37 2024

@author: cornelius
"""


# ---------------------------- IMPORTE ------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# ------------------------- DATENVORBEREITUNG --------------------------

df = pd.read_csv("star_classification.csv")

label_encoder = LabelEncoder()
df["class"] = label_encoder.fit_transform(df["class"])

print(list(set(df["class"].values.tolist())))
print(df.info())
features = df.drop(columns=["class","obj_ID","alpha","delta"])
labels = df["class"]

feature_train, feature_test, label_train, label_test = \
                        train_test_split(features,labels,test_size=0.2,random_state=37)
                        
feature_train = np.array(feature_train.values.tolist())
feature_test = np.array(feature_test.values.tolist())
label_train = label_train.values.tolist()
label_test = label_test.values.tolist()

# Normalize Data

x = df.values
min_max = preprocessing.StandardScaler()
x_scaled = min_max.fit_transform(x)
df = pd.DataFrame(x_scaled)



# ------------------------- NEURAL NETWORK -------------------------------



input_nodes = 14
hidden_nodes = 10
output_nodes = 3
learning_rate = 1e-2
epochs = 4





w_ih = np.random.rand(hidden_nodes,input_nodes)
w_ho = np.random.rand(output_nodes,hidden_nodes)

# feedforward(): Signalübertragung
# training(): Signalübertragung, Fehler berechnet, Gewichte aktualisiert


def feedforward(input_list,w_ih,w_ho):
    
    input_list = np.array(input_list,ndmin=2)
    input_hidden = input_list @ w_ih.T
    output_hidden = scipy.special.expit(input_hidden)
    
    input_final = output_hidden @ w_ho.T
    output_final = scipy.special.expit(input_final)
    
    return output_final






def train(input_list,w_ih,w_ho, target_list,learning_rate):
    target_list = np.array(target_list,ndmin=2) # Neu
    input_list = np.array(input_list,ndmin=2)
    input_hidden = input_list @ w_ih.T
    output_hidden = scipy.special.expit(input_hidden)
    input_final = output_hidden @ w_ho.T
    output_final = scipy.special.expit(input_final)
    
    
    output_errors = target_list - output_final
    hidden_errors = output_errors @ w_ho
    
    w_ho += learning_rate*np.dot((output_errors*output_final*(1-output_final)).T,output_hidden)
    w_ih += learning_rate*np.dot((hidden_errors*output_hidden*(1-output_hidden)).T,input_list)
    
  
    
  
# ------------------------------- TRAINING ------------------------------  


  
for e in range(epochs):
    for i in range(len(feature_train)):
        target_vector = np.zeros(output_nodes)
        target_vector[int(label_train[i])] = 0.99
        train(feature_train[i],w_ih,w_ho,target_vector,learning_rate)
    
        if (i%1000)==0:
            print("Training")





# ------------------------------- TESTING ------------------------------


def accuracy(feature_test,label_test,w_ih,w_ho):
    
    count = 0 # Count für alle richtigen
    
    for i in range(len(feature_test)):
        prediction = np.argmax(feedforward(feature_test[i], w_ih, w_ho))
        tatsache = label_test[i]
        if prediction == tatsache:
            count+=1
    
    return count/len(feature_test)



print(accuracy(feature_test,label_test,w_ih,w_ho))
    
    








           