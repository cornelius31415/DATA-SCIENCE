#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:20:53 2024

@author: cornelius
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 21:46:33 2022

@author: cornelius
"""

import statistics

liste = ["groß","klein","mittel","mittel","mittel","mittel","mittel","groß"]
liste2 = [1,2,2,2,3,4,5,5,5,5,5,7,7]




def modus(data):
    anzahl = []
    trimmed = set(data)
    trimmed_list = list(trimmed)
    
    
    for i in range(0,len(trimmed_list)):
        #zähle wie oft jedes Element vorkommt
        x = data.count(trimmed_list[i])
        #füge die Elemente zur Liste anzahl zu
        anzahl.append(x)
        
    max_anzahl = max(anzahl)
    modus = trimmed_list[anzahl.index(max_anzahl)]
        
    
    return modus
    

print(modus(liste))
print(modus(liste2))




print(statistics.mode(liste))
print(statistics.mode(liste2))