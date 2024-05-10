#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:45:50 2024

@author: cornelius
"""

# ----------------------------------------------------------------------------

# -----------------       STELLAR CLASSIFICATION         ---------------------

# ----------------------------------------------------------------------------




print("----------------------------------------------------------------------------")
print()
print(" -----------------       STELLAR CLASSIFICATION         ---------------------")
print()
print("----------------------------------------------------------------------------")




# ---------------------------- IMPORTS ---------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.special
from NeuralNet import NeuralNetwork




# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("star_classification.csv")
df.drop(columns=['obj_ID'])
df.dropna()
print(df.info())

label_encoder = LabelEncoder()
df["class"] = label_encoder.fit_transform(df["class"])

features = df.drop(columns=["class"])
labels   = df['class']

feature_train, feature_test, label_train, label_test = train_test_split(
    features,labels, test_size=0.2, random_state=42)






# ----------------------------------------------------------------------------
#                             DECISION TREE   
# ----------------------------------------------------------------------------


decision_tree = DecisionTreeClassifier()
decision_tree.fit(feature_train,label_train)
prediction = decision_tree.predict(feature_test)


conf_matrix = confusion_matrix(label_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, 
                                            display_labels = [0,1,2])
cm_display.plot()
plt.show()


accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction,average='weighted')
sensitivity = metrics.recall_score(label_test, prediction,average='weighted')
f1_score = metrics.f1_score(label_test, prediction,average='weighted')

print()
print("----------------------- Decision Tree Results -----------------------")
print()

print("The accuracy is: ", accuracy)
print("The precision is: ", precision)
print("The sensitivity is: ", sensitivity)
print("The F1 Score is: ", f1_score)

"""
                            OUTPUT
                    
               The accuracy is:  0.96465
               The precision is:  0.9647081456158808
               The sensitivity is:  0.96465
               The F1 Score is:  0.9646778479183328 
"""


# ----------------------------------------------------------------------------
#                             K NEAREST NEIGHBORS   
# ----------------------------------------------------------------------------
# pip install threadpoolctl==3.1.0

knn = KNeighborsClassifier()
knn.fit(feature_train, label_train)

prediction = knn.predict(feature_test)



conf_matrix = confusion_matrix(label_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, 
                                            display_labels = [0,1,2])
cm_display.plot()
plt.show()


accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction,average='weighted')
sensitivity = metrics.recall_score(label_test, prediction,average='weighted')
f1_score = metrics.f1_score(label_test, prediction,average='weighted')

print()
print("----------------------- KNN Results -----------------------")
print()

print("The accuracy is: ", accuracy)
print("The precision is: ", precision)
print("The sensitivity is: ", sensitivity)
print("The F1 Score is: ", f1_score)



# ----------------------------------------------------------------------------
#                               NEURAL NETWORK
# ----------------------------------------------------------------------------

        

input_nodes = 17
hidden_nodes = 10
output_nodes = 3
learning_rate = 1e-2
epochs = 1

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

n.fit(feature_train,label_train,epochs)

prediction = n.predict(feature_test)




conf_matrix = confusion_matrix(label_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, 
                                            display_labels = [0,1,2])
cm_display.plot()
plt.show()


accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction,average='weighted')
sensitivity = metrics.recall_score(label_test, prediction,average='weighted')
f1_score = metrics.f1_score(label_test, prediction,average='weighted')

print()
print("----------------------- NEURAL NETWORK Results -----------------------")
print()

print("The accuracy is: ", accuracy)
print("The precision is: ", precision)
print("The sensitivity is: ", sensitivity)
print("The F1 Score is: ", f1_score)










































