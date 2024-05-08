#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:20:17 2024

@author: cornelius
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 21:46:05 2022

@author: cornelius
"""

import statistics as st

liste = [1,2,2,3,4,5,6,6,7,8,2,54,2,1,2,3]

def median_function(data):
    sorted_data = sorted(data)
    data_list = list(sorted_data)
    n = len(data)
    index = n // 2
    
    if n%2:
        return data_list[index]
    return sum(data_list[index-1:index+1])/2
   
    
    pass



print(median_function(liste))
print(st.median(liste))