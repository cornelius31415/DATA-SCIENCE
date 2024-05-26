#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:51:21 2024

@author: cornelius
"""





"""

    predicting the price of a stock 50 days in the future - starting 50 days ago until now

                            Markov Chain Stock Analysis
                
                ticker names: 
                    
                    SPY          S&P 500
                    AAPL         Apple
                    ABBV         AbbVie (window size 100)
                    CL           Colgate Palmolive
                    DLAKY        Lufthansa
                    KMB          Kimberly Clark
                    BTI          British American Tobacco
                    BFFAF        BASF
                    RWNFF        RWE
                    VODPF        Vodafone
   
                Further tickers: https://stockanalysis.com/stocks/



"""

# ---------------------------- IMPORTS ------------------------------------

import yfinance as yf 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




# ---------------------------- Data Preparation ---------------------------------

ticker = "ABBV" 
#ticker = "AAPL" 
#ticker = "ABBV" 
#ticker = "CL" 
#ticker = "DLAKY" 
#ticker = "KMB" 
#ticker = "BTI" 









data = yf.download(ticker,start="2020-12-31", end = "2024-04-27")

data["daily_return"] = data["Adj Close"].pct_change()
data['state'] = np.where(data["daily_return"]>0,"up","down")
data.dropna()
print(data.info())
# data.to_excel("SPY.xlsx")

# data['state'] = np.zeros(len(data["daily_return"]))

close = data["Adj Close"].tolist()



# --------- Gruppierung in u1, u2, u3, d1, d2, d3 Kategorien -----------------
new_states = []
print(len(close))
values = data["daily_return"].tolist()
print(len(values))
values = [value*100 for value in values]
values[0] = 0

for value in values:
    if 0 < value < .5:
        new_states.append("u1")
    elif .5 <= value < 1:
        new_states.append("u2")
    elif value >= 1:
        new_states.append("u3")
    elif -.5 < value < 0: 
        new_states.append("d1")
    elif -1 < value < 0.5:
        new_states.append("d2")
    elif value <= -1:
        new_states.append("d3")

#-----------------------------------------------------------------------------


data["new_state"] = new_states
cat6 = data["new_state"].tolist()
cat6 = list(set(cat6))

states = data['new_state'].tolist()
percentages = data["daily_return"].tolist()
percentages = list(zip(states,percentages))
print()

# -------------------------------------------------------------------------------





# Make a list of 2-tuples out of a list

def make_tupel_list(liste):
    
    tupel_list = [(liste[i],liste[i+1]) for i in range(0,len(liste)-1)]
    return tupel_list




def transmatrix(states):
    
    # determine the categories
    categories = list(set(states))
    
    # create a list of transition tuples
    state_transitions = make_tupel_list(states)
    
    # initialize transition matrix as dataframe with zeros
    transmat = pd.DataFrame(0,index=categories,columns=categories)
    
    # double iteration variable for iterating through tuples
    for i, j in state_transitions:
        transmat[i][j] += 1
        
    # sum of each row
    row_sum = transmat.sum(axis=1)
    
    # divide each element by sum of row
    transition_matrix = transmat.div(row_sum,axis=0).fillna(0.00)
    
    
    return transition_matrix.to_numpy()


            
categories = ["u1","u2","u3","d1","d2","d3"]    


# -----------------------------------------------------------------------------


# average percentage change per category

def average(categories,data,states):
    #category averages: a list of percentages, each element corresponds to a category
    cat_averages = []
    
    for category in categories:
        # take those daily return values that are labelled with required category
        cat_values = data.loc[data[states]==category,"daily_return"]
        cat_averages.append(np.mean(cat_values))
        
    return cat_averages



cat6_average = average(categories,data,"new_state")


# the amount that gets added to today's price: the predicted price change
def predicted_change(states,cat_averages,trans_mat):
    
    categories = list(set(states))
    
    state = states[-1]
    state_vector = np.zeros(len(categories))
    

    
    for idx, category in enumerate(categories):
        if category == state:
            state_vector[idx]=1
    
    prob_vector = state_vector @ trans_mat
    prob_vector = prob_vector.tolist()

        

    prediction = cat_averages[prob_vector.index(max(prob_vector))]

    return prediction
            
     





# Prediction for a window

def clairvoyant(window,window2, categories, closes):
    
    window = window
    trans_mat = transmatrix(window)
    
    pred = predicted_change(window, cat6_average, trans_mat)



    consecutive_days = 3
    last_states = window2[len(window2)-consecutive_days:len(window2)]
    
    # check if all elements in last_states are the same
    boolsch = True
    i=0
    while i < len(last_states)-1:
        if last_states[i]== last_states[i+1]:
            boolsch = True
            i+=1
        else:
            boolsch = False
            break 
    prediction_parameter = 1
    
    if boolsch == True:
        day0 = window[-1]
        prediction = (-1)*closes[-1]*pred*prediction_parameter + closes[-1]
    elif boolsch == False:
        day0 = window[-1]
        prediction = closes[-1]*pred*prediction_parameter + closes[-1]
       
       
    

    
    return prediction



start =800


window2 = data['state'].tolist()
#print(states[start-200:start])
print()
print(data.iloc[start-1])


predicted_data = []
for i in range(0,start):
    predicted_data.append(close[i])

for i in range(start,len(close)):

    predicted_data.append(clairvoyant(states[i-50:i],window2[i-50:i], categories, predicted_data))




df = pd.DataFrame({'Real Price':close[750:834]})
df['Predicted Prices']=predicted_data[750:834]
df['Real Price'].plot()
df['Predicted Prices'].plot()
plt.legend()
plt.ylabel("Price")
plt.title(ticker)
plt.show()
