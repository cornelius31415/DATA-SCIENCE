#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:19:49 2024

@author: cornelius
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 21:46:44 2022

@author: cornelius
"""

liste = [12,4,23,5,3,2,533,53,2]

def mean(data):
    
    mean = sum(data)/len(data)
    return mean
    
print(mean(liste))





import statistics as st
print(st.mean(liste))