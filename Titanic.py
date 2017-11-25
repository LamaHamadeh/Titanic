#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:08:20 2017

@author: lamahamadeh
"""

import pandas as pd
import matplotlib.pyplot as plt

#Download the train dataset
#--------------------------
Data = pd.read_csv('/Users/lamahamadeh/Desktop/Python/Titanic/train.csv')

#================================
#Exploratory Data Analysis (EDA)
#================================

#analyse the train data
#-----------------------
print(Data.head()) #take a look at the data
print(Data.shape) #(891, 12)
print(Data.describe()) #describe only shows the numerical data in the dataset

#Checking the number/percentage of the survived passengers
print(Data.Survived.value_counts()) #0:549  #1:342
print('The percentage of the survived people on the Titanic is', 342/891.0) #38%
print('The percentage of the not-survived people on the Titanic is', 549/891.0) #61%

#checking the sex/percentage of the passengers
print(Data.Sex.value_counts()) #Male:577  #Female:314
print('The percentage of the male passengers on the Titanic is', 577/891.0) #64%
print('The percentage of the female passengers on the Titanic is', 314/891.0) #35%

#check for Nans
def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print ("Missing values per column:")
print (Data.apply(num_missing, axis=0)) #177 nans in the 'age' feature
#and 687 nans in the 'cabin' feature. 

#fill the 177 nans in the Age columns with the mean of age of the whole dataset.
Data['Age'].fillna(Data.Age.mean(), inplace = True) 

#drop the Cabin, Embarked and Name coloumns as they don't do any good for our dataset
Data = Data.drop(['Cabin', 'Embarked', 'Name'], axis = 1)


#Visulaisation
#-------------
#plot the sex of the passengers (categorical data)
plt.figure(1)
Data.Sex.value_counts().plot(kind='bar', color='#6878cc', edgecolor = '#6878cc') #visualise the sex column of the data

#plot the distribution ofthe ticket fare (numerical data)
plt.figure(2)
Data.Fare.hist(color='#00B28C', edgecolor = '#00B28C')
plt.xlabel('Ticket Fare')
plt.ylabel('Number of People')

#compare between male, femal and children passengers: survived or not
Male_Survived = Data[Data.Sex =='male'].Survived.value_counts() #Survived male sub_dataset
Female_Survived = Data[Data.Sex == 'female'].Survived.value_counts() #Survived female sub_dataset
Children_Survived = Data[Data.Age < 15].Survived.value_counts() #Survived children sub_dataset
fig, axs = plt.subplots(1,3)
Male_Survived.plot(kind='barh', color = '#024dce',  edgecolor = '#024dce', title = 'Male Survivorship', ax = axs[0])
Female_Survived.plot(kind = 'barh', color = '#df3fd0', edgecolor = '#df3fd0', title = 'Female Survivorship', ax = axs[1])
Children_Survived.plot(kind = 'barh', color = '#e2e35d', edgecolor = '#e2e35d', title = 'Children (<15) Survivorship', ax = axs[2])
                     

#plot a kernel density estimate of the subset of the 1st class passenger's age
plt.figure(4)
Data.Age[Data.Pclass == 1].plot(kind = 'kde', color = 'red')
Data.Age[Data.Pclass == 2].plot(kind = 'kde', color = 'blue')
Data.Age[Data.Pclass == 3].plot(kind = 'kde', color = 'green')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age distribution Within Classes')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc = 'best')

#visulaise the distribution of the passengers's age 
plt.figure(5)
Data.Age.hist(color= '#C8515F', edgecolor = '#9f3e4a')
plt.xlabel('Age')
plt.ylabel('Count')






















plt.show()
