#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:19:41 2024

@author: cornelius
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 23:16:58 2022

@author: cornelius
"""

x = [1.6,2.6,3.7,5.3,6.9,7.1,7.2,6.7,5.1,3.6,2.1,1.4]
y = [28300,28000,41000,40000,48000,47500,43000,50700,50000,48000,25000,24000]


import statistics as st
import numpy as np

def kovarianz(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    
    summanden = []
    
    for i in range(0,len(x)):
        summand = (x[i]-mean_x)*(y[i]-mean_y)
        summanden.append(summand)
        kovarianz = sum(summanden)/(len(y)-1)
        pass
    return kovarianz
    
 

    
    

def korrelation(x,y):
    
    cov = kovarianz(x,y)
    std_x = np.std(x)
    print(std_x)
    std_y = np.std(y)
    print(std_y)
    korrelation = cov/(std_x*std_y)
    return korrelation
    

print(korrelation(x,y))