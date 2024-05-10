#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:37:52 2024

@author: cornelius
"""


# ---------------------------- IMPORTS ---------------------------------
from sklearn.naive_bayes import GaussianNB

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt



# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("mushroom.csv")
pd.set_option('display.max_columns', None)

features = df.drop(columns="class")
labels   = df['class']



feature_train, feature_test, label_train, label_test = train_test_split(
    features,labels, test_size=0.2, random_state=42)




# --------------------    Naives Bayes   -------------------------------

naive = GaussianNB()
naive.fit(feature_train,label_train)
prediction = naive.predict(feature_test)



accuracy = metrics.accuracy_score(label_test, prediction)
precision = metrics.precision_score(label_test, prediction)
sensitivity = metrics.recall_score(label_test, prediction)
f1_score = metrics.f1_score(label_test, prediction)

"""
1. Accuracy:        What percentage of shrooms has been correctly identified
2. Precision:       What percentage of predicted poisonous shrooms is actually poisonous
3. Recall:          What percentage of the poisonous shrooms is correctly identified as poisonous.
4. F1 Score:        The balance between precision and recall
"""

print("The accuracy of the decision tree is: ", accuracy)
print("The precision of the decision tree is: ", precision)
print("The sensitivity of the decision tree is: ", sensitivity)
print("The F1 Score of the decision tree is: ", f1_score)
