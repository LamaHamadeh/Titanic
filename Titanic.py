#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:08:20 2017

@author: lamahamadeh
"""
#Importing necessary libraries
#--------------------------
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression 

#Download the train dataset
#--------------------------
Data_train = pd.read_csv('/Users/lamahamadeh/Desktop/Python/Titanic/train.csv')

#================================
#Exploratory Data Analysis (EDA)
#================================

#analyse the train data
print(Data_train.head()) #take a look at the data
print(Data_train.shape) #(891, 12)
print(Data_train.describe()) #describe only shows the numerical data in the dataset

#Checking the number/percentage of the survived passengers
print(Data_train.Survived.value_counts()) #0:549  #1:342
print('The percentage of the survived people on the Titanic is', 342/891.0) #38%
print('The percentage of the not-survived people on the Titanic is', 549/891.0) #61%

#checking the sex/percentage of the passengers
print(Data_train.Sex.value_counts()) #Male:577  #Female:314
print('The percentage of the male passengers on the Titanic is', 577/891.0) #64%
print('The percentage of the female passengers on the Titanic is', 314/891.0) #35%

#check for Nans
def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print ("Missing values per column:")
print (Data_train.apply(num_missing, axis=0)) #177 nans in the 'age' feature
#and 687 nans in the 'cabin' feature. 

#fill the 177 nans in the Age feature with the mean value of all ages
#and 687 nans in the 'cabin' feature with 0
Data_train['Age'].fillna(Data_train.Age.mean(), inplace = True)
Data_train.drop(['Cabin', 'Ticket'], inplace = True, axis = 1)

#Visulaisation
#plot the sex of the passengers (categorical data)
plt.figure(1)
Data_train.Sex.value_counts().plot(kind='bar', color='#6878cc', edgecolor = '#6878cc') #visualise the sex column of the data

#plot the distribution ofthe ticket fare (numerical data)
plt.figure(2)
Data_train.Fare.hist(color='#00B28C', edgecolor = '#00B28C')
plt.xlabel('Ticket Fare')
plt.ylabel('Number of People')

#compare between male, femal and children passengers: survived or not
Male_Survived = Data_train[Data_train.Sex =='male'].Survived.value_counts() #Survived male sub_dataset
Female_Survived = Data_train[Data_train.Sex == 'female'].Survived.value_counts() #Survived female sub_dataset
Children_Survived = Data_train[Data_train.Age < 15].Survived.value_counts() #Survived children sub_dataset
fig, axs = plt.subplots(1,3)
Male_Survived.plot(kind='barh', color = '#024dce',  edgecolor = '#024dce', title = 'Male Survivorship', ax = axs[0])
Female_Survived.plot(kind = 'barh', color = '#df3fd0', edgecolor = '#df3fd0', title = 'Female Survivorship', ax = axs[1])
Children_Survived.plot(kind = 'barh', color = '#e2e35d', edgecolor = '#e2e35d', title = 'Children (<15) Survivorship', ax = axs[2])
                     
#plot a kernel density estimate of the subset of the 1st class passenger's age
plt.figure(4)
Data_train.Age[Data_train.Pclass == 1].plot(kind = 'kde', color = 'red')
Data_train.Age[Data_train.Pclass == 2].plot(kind = 'kde', color = 'blue')
Data_train.Age[Data_train.Pclass == 3].plot(kind = 'kde', color = 'green')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age distribution Within Classes')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc = 'best')

#visulaise the distribution of the passengers's age 
plt.figure(5)
Data_train.Age.hist(color= '#C8515F', edgecolor = '#9f3e4a')
plt.xlabel('Age')
plt.ylabel('Count')

#======================================
#Clustering using kmean method (Kmeans)
#======================================

#first, let's handle the non-numeric data in the dataset
print(Data_train.dtypes)
#study only the numeric data
numeric_values = list(Data_train.dtypes[Data_train.dtypes != 'object'].index)
print(Data_train[numeric_values].head())

#Create the kmeans model on the data
X = np.array(Data_train[numeric_values].drop(['Survived'], axis=1))
X = preprocessing.scale(X) #this step increases the rprediction resolution of the model
y = np.array(Data_train['Survived'])
kmeans_model = KMeans(n_clusters = 2, init = 'random', n_init = 60, max_iter = 400, random_state = 43)
kmeans_model.fit(X)
centroids = kmeans_model.cluster_centers_
KM_labels = kmeans_model.labels_ #these are the labels generated from the KMeans clustering method
OR_labels = Data_train.Survived #these are the original labels provided by the dataset
Data_train['kMean predicted label'] = kmeans_model.labels_ #Here we add to the dataset a column that 
#contains the KMeans prediction labels in order to compare between the Survived column and the kmean clustering perdication
print (Data_train.head())

#predict the resolution of the model
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans_model.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))
#without the preprocessing step, the prediction value is about 48%
#however, after inserting this step, the prediction becomes 0.71% which is much better

#import the test dataset
Data_test = pd.read_csv('/Users/lamahamadeh/Desktop/Python/Titanic/test.csv')

Data_test['Survived'] =1.23

def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print ("Missing values per column:")
print (Data_test.apply(num_missing, axis=0)) #86 in Age, 1 in Fare and 
#327 in Cabin

Data_test['Age'].fillna(Data_test.Age.mean(), inplace = True)
Data_test['Fare'].fillna(Data_test.Fare.mean(), inplace = True)

numeric_values_test = list(Data_test.dtypes[Data_test.dtypes != 'object'].index)

print(Data_test[numeric_values_test].head())

#prediction = kmeans_model.predict(Data_test[numeric_values_test])



#======================================================
#Classification using k nearest neighbours method (Knn)
#======================================================

numeric_values = list(Data_train.dtypes[Data_train.dtypes != 'object'].index)

print(Data_train[numeric_values].head())
X = preprocessing.scale(X)
X = np.array(Data_train[numeric_values].drop(['Survived'], axis=1))

y = np.array(Data_train['Survived'])

#cross_validation
X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size=0.5, random_state = 7)

#apply the knn method
Knn = KNeighborsClassifier(n_neighbors = 2)
#train the data
Knn.fit(X_train,Y_train)
#test the accuracy of the data
accuracy = Knn.score(X_test, Y_test) #the accuracy of the model is 66%
#without the preprocessing the accuracy drops to 60%

print('accuracy of the model is: ', accuracy)

#import the test dataset
Data_test = pd.read_csv('/Users/lamahamadeh/Desktop/Python/Titanic/test.csv')

Data_test['Survived'] =1.23

def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print ("Missing values per column:")
print (Data_test.apply(num_missing, axis=0)) #86 in Age, 1 in Fare and 
#327 in Cabin

Data_test['Age'].fillna(Data_test.Age.mean(), inplace = True)
Data_test['Fare'].fillna(Data_test.Fare.mean(), inplace = True)

numeric_values_test = list(Data_test.dtypes[Data_test.dtypes != 'object'].index)

print(Data_test[numeric_values_test].head())



#==================================
#Prediction using Linear regression
#==================================













