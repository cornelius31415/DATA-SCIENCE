#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:32:05 2024

@author: cornelius
"""

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from NeuralNetworkClass import NeuralNetwork
import DeepNNClass as D



df = pd.read_csv("sentiment_analysis.csv")

# df = df.sample(n=5000, random_state=42)
# df = df.head(1000)
# df.to_csv('sentiment_analysis.csv', index=False)


# Data Cleaning
df = df.dropna(subset=['text'])


label_encoder = LabelEncoder()
df["sentiment"] = label_encoder.fit_transform(df["sentiment"])

text = df['text']
sentiment = df['sentiment']

num_labels = df['sentiment'].nunique()

# # ------------------------ BAG OF WORDS DATAFRAME  ---------------------------

sentences = text.values.tolist()


count_vec = CountVectorizer()
word_counts = count_vec.fit_transform(sentences)
bag_of_words_df = pd.DataFrame(word_counts.toarray(),columns = count_vec.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(bag_of_words_df,sentiment,
                                                    test_size=0.2, random_state=42)

# # ---------------------------    DECISION TREE    ----------------------------



decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
prediction = decision_tree.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print(f"\n                  Decision Tree Accuracy: {accuracy} \n")
print("\n")

# # ---------------------------    NEURAL NETWORK    ----------------------------
# #                                   3 layers

# nn = NeuralNetwork(X_train.shape[1], 100, num_labels, 1e-2)
# nn.fit(X_train, y_train, 30)
# prediction = nn.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, prediction)
# print(accuracy)



# # ---------------------------    DEEP NEURAL NETWORK    ----------------------------


n = D.NeuralNetwork(X_train.shape[1], 1e-1)
n.layer(100)
n.layer(100)
n.layer(100)
n.layer(num_labels)

X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

n.fit(X_train, y_train, 20)
prediction = n.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print(accuracy)


















