#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:13:48 2024

@author: cornelius
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 09:46:22 2022

@author: cornelius
"""
import math


def entropy(data):
    #zähle wie oft jedes Element vorkommt
    new = set(data)
    new_data = list(new)
    anzahl = []
    probability_list = []
    for i in range(0,len(new_data)):
        x = data.count(new_data[i])
        anzahl.append(x)
        #bestimme die relative Häufigkeit jedes Elements
        probability_list.append(anzahl[i]/len(data))
        pass
    #summiere über die einzelnen Algorithmen
    entropy = - sum(p*math.log2(p) for p in probability_list)
    return entropy





def information(data_before, data_after):
    gain = entropy(data_before)-entropy(data_after)
    return gain 


def filter_set(data, search_string):
	def iterator_func(x):
		for v in x.values():
			if search_string in v:
				return True
		return False
	return filter(iterator_func, data)




def information_gain(klasse,feature):
    feature_categories = list(set(feature))
    anzahl = []
    probability_list = []
    
    for i in range(0,len(feature_categories)):
        x = feature.count(feature_categories[i])
        anzahl.append(x)
        probability_list.append(anzahl[i]/len(feature))
    
    pairs = []
    for i in range(0,len(feature)):
        dic = {"feature":str(feature[i]),"class":klasse[i]}
        pairs.append(dic)
     
    summanden = []   
    for i in range(0,len(feature_categories)):
        
        filtered_data = list(filter_set(pairs,str(feature_categories[i])))
        
        after_event = []
        for n in range(0,len(filtered_data)):
            x = filtered_data[n]['class']
            after_event.append(x)
        
        info = information(klasse,after_event)
        gain = probability_list[i]*info
        summanden.append(gain)
        
    information_gain = sum(l for l in summanden)
    return information_gain
        