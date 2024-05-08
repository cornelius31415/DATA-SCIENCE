#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:43:57 2024

@author: cornelius
"""

# ---------------------------- IMPORTS ---------------------------------------

from sklearn.linear_model import LogisticRegression

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


# --------------------   LOGISTIC REGRESSION   -------------------------------

log_regression = LogisticRegression()
log_regression.fit(feature_train,label_train)

prediction = log_regression.predict(feature_test)


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

print("The accuracy of the decision tree is: ", accuracy)
print("The precision of the decision tree is: ", precision)
print("The sensitivity of the decision tree is: ", sensitivity)
print("The F1 Score of the decision tree is: ", f1_score)



"""
                                OUTPUT
                
        The accuracy of the decision tree is:  0.6417468541820873
        The precision of the decision tree is:  0.6601123595505618
        The sensitivity of the decision tree is:  0.7140445644834571
        The F1 Score of the decision tree is:  0.6860201102821927

"""



