#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:24:15 2024

@author: cornelius
"""


# ---------------------------- IMPORTS ---------------------------------
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("emails.csv")
#pd.set_option('display.max_columns', None)



features = df.drop(columns=["Prediction","Email No."])
labels   = df['Prediction']

feature_train, feature_test, label_train, label_test = train_test_split(
    features,labels, test_size=0.2, random_state=42)

# --------------------   DECISION TREE   -------------------------------


decision_tree = DecisionTreeClassifier()
decision_tree.fit(feature_train,label_train)
prediction = decision_tree.predict(feature_test)


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
                
              The accuracy is:  0.9188405797101449
              The precision is:  0.8581081081081081
              The sensitivity is:  0.8581081081081081
              The F1 Score is:  0.8581081081081081

"""