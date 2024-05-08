#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:16:47 2024

@author: cornelius
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 09:52:18 2022

@author: cornelius
"""


import pandas as pd
import math
from InformationGain import entropy, information_gain
import numpy as np
import statistics as st


#Importieren der Daten
#Jedes Tabellenblatt wird in ein Dataframe gepackt
file = 'ProjectData.xlsx'
xl = pd.ExcelFile(file)
df1 = xl.parse("Tabellenblatt1")
df2 = xl.parse("Tabellenblatt2")

#print(df1.to_string())
#print(df2.to_string())

#Tabellenblatt 1: Die Spalten in Listen packen
wetteraussicht= df1.get("Wetteraussicht").values.tolist()
temperaturkategorie = df1.get("Temperaturkategorie").values.tolist()
humidkategorie = df1.get("Luftfeuchtigkeit").values.tolist()
windstaerke = df1.get("Windstärke").values.tolist()
draussen_essen = df1.get("Draussen Essen").values.tolist()

print(windstaerke)


#Tabellenblatt 2: Die Spalten in Listen packen
temperatur = df2.get("Temperatur").values.tolist()
humidity = df2.get("Luftfeuchtigkeit").values.tolist()
wind_speed = df2.get("Windgeschwindigkeit").values.tolist()
aussenverkauf = df2.get("Aussenverkauf").values.tolist()



#Berechne den Informationsgewinn der Features aus Tabellenblatt 1
print("Temperatur Informationsgewinn:",round(information_gain(draussen_essen, temperaturkategorie),4))   
print("Wetteraussicht Informationsgewinn:",round(information_gain(draussen_essen, wetteraussicht),4))
print("Luftfeuchtigkeit Informationsgewinn:",round(information_gain(draussen_essen, humidkategorie),4))
print("Windstärke Informationsgewinn:",round(information_gain(draussen_essen,windstaerke),4))
print(" ")




#Kategorisierung der numerischen Daten

temperatur_kategorien = []
humidity_kategorien = []
wind_kategorien = []

#Kategorisierung der Temperaturdaten
for x in temperatur:
    
    if x<18:
        temperatur_kategorien.append("kalt")
    elif x>18 and x<28:
        temperatur_kategorien.append("mild")
    elif x>28:
        temperatur_kategorien.append("heiss")

#Kategorisierung der Windgeschwindigkeiten
for x in wind_speed:
    
    if x<5:
        wind_kategorien.append("1")
    elif x>5 and x<11:
        wind_kategorien.append("2")
    elif x>11 and x<19:
        wind_kategorien.append("3")
        

#Kategorisierung der Luftfeuchtigkeit
for x in humidity:
    if x > 5:
        humidity_kategorien.append("hoch")
    else:
        humidity_kategorien.append("niedrig")
        

#Berechne den Informationsgewinn von jedem Feature mit Bezug auf den Aussenverkauf    
print("Temperatur Informationsgewinn:",round(information_gain(aussenverkauf, temperatur_kategorien),4))   
print("Wind Informationsgewinn:",round(information_gain(aussenverkauf, wind_kategorien),4))
print("Luftfeuchtigkeit Informationsgewinn:",round(information_gain(aussenverkauf, humidity_kategorien),4))
print(" ")

#Statistikfunktionen
def kovarianz(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    
    summanden = []
    
    for i in range(0,len(x)):
        summand = (x[i]-mean_x)*(y[i]-mean_y)
        summanden.append(summand)
        kovarianz = sum(summanden)/(len(y))
        pass
    return kovarianz     

def korrelation(x,y):
    
    cov = kovarianz(x,y)
    std_x = np.std(x)
    #print(std_x)
    std_y = np.std(y)
    #print(std_y)
    korrelation = cov/(std_x*std_y)
    return korrelation
    



#Berechne die Mittelwerte der Features
print("Die mittlere Temperatur beträgt:",st.mean(temperatur))
print("Die mittlere Luftfeuchtigkeit beträgt:",st.mean(humidity))
print("Die mittlere Windgeschwindigkeit beträgt:",st.mean(wind_speed))
print(" ")


#Berechne den Median der Features
print("Der Median der Temperatur ist:",st.median(temperatur))
print("Der Median der Luftfeuchtigkeit ist:",st.median(humidity))
print("Der Median der Windgeschwindigkeit ist:",st.median(wind_speed))
print(" ")



#Berechne die Standardabweichung der Features
print("Die Standardabweichung der Temperatur beträgt:",np.std(temperatur))
print("Die Standardabweichung Luftfeuchtigkeit beträgt:",np.std(humidity))
print("Die Standardabweichung Windgeschwindigkeit beträgt:",np.std(wind_speed))
print(" ")




#Berechne die Korrelation zwischen der Luftfeuchtigkeit und der Temperatur
print("Korrelation Luftfeuchtigkeit und Temperatur",korrelation(humidity, temperatur))


#Berechne die Korrelation zwischen der Luftfeuchtigkeit und der der Windgeschwindigkeit
print("Korrelation Luftfeuchtigkeit und Windgeschwindigkeit",korrelation(humidity, wind_speed))


#Berechne die Korrelation zwischen der Windgeschwindikeit und dem Temperatur
print("Korrelation Windgeschwindigkeit und Temperatur",korrelation(wind_speed,temperatur))