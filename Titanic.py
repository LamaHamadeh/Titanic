#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:08:20 2017

@author: lamahamadeh
"""

import pandas as pd
import matplotlib.pyplot as plt

#Download the train dataset
Data = pd.read_csv('/Users/lamahamadeh/Desktop/Python/Titanic/train.csv')

#analyse the data
print(Data.head()) #take a look at the data
print(Data.shape) #(891, 12)
print(Data.describe())
print(Data.Survived.value_counts()) #0:549  #1:342
print('The percentage of the survived people on the Titanic is', 342/891.0) #38%
print('The percentage of the not-survived people on the Titanic is', 549/891.0) #61%

print(Data.Sex.value_counts()) #Male:577  #Female:314
print('The percentage of the male passengers on the Titanic is', 577/891.0) #64%
print('The percentage of the female passengers on the Titanic is', 314/891.0) #35%
plt.figure(1)
Data.Sex.value_counts().plot(kind='bar', color='#6878cc') #visualise the sex column of the data
#we can notice here that out data is catagorical!

plt.figure(2)
Data.Fare.hist(color='#00B28C')
plt.xlabel('Ticket Fare')
plt.ylabel('Number of People')


Male_Survived = Data[Data.Sex =='male'].Survived.value_counts()
Female_Survived = Data[Data.Sex == 'female'].Survived.value_counts()
Children_Survived = Data[Data.Age < 15].Survived.value_counts()
plt.figure(3)

fig, axs = plt.subplots(1,3)
Male_Survived.plot(kind='barh', color = '#024dce', title = 'Male Survivorship', ax = axs[0])
Female_Survived.plot(kind = 'barh', color = '#df3fd0', title = 'Female Survivorship', ax = axs[1])
Children_Survived.plot(kind = 'barh', color = '#e2e35d', title = 'Children (<15) Survivorship', ax = axs[2])
                     


