#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:50 2024

@author: cornelius
"""

# ---------------------------- IMPORTS ---------------------------------

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# ------------------------- DATA PREPARATION ---------------------------------

df = pd.read_csv("mushroom.csv")
pd.set_option('display.max_columns', None)

features = df.drop(columns="class")
labels   = df['class']

feature_train, feature_test, label_train, label_test = train_test_split(
    features,labels, test_size=0.1, random_state=42)


# --------------------   K NEAREST NEIGHBORS   -------------------------------



knn = KNeighborsClassifier()
knn.fit(feature_train, label_train)

prediction = knn.predict(feature_test)


# ------------------- CONFUSION MATRIX ---------------------------------------


conf_matrix = confusion_matrix(label_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, 
                                            display_labels = [0,1])
cm_display.plot()
plt.show()


accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction)
sensitivity = metrics.recall_score(label_test, prediction)
f1_score = metrics.f1_score(label_test, prediction)

print("The accuracy is: ", accuracy)
print("The precision is: ", precision)
print("The sensitivity is: ", sensitivity)
print("The F1 Score is: ", f1_score)


"""
                                OUTPUT
                
              The accuracy is:  0.7209474463360473
              The precision is:  0.7515570934256055
              The sensitivity is:  0.7332883187035787
              The F1 Score is:  0.7423103212576897

"""