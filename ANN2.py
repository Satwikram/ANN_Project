# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:26:01 2020

@author: SATWIK RAM K
"""

#importing libraries
import numpy as np
import pandas as pd
import keras

#importing datasets
dataset = pd.read_csv('Churn_Modelling.csv')

#Working on Dummy Variables
df = pd.get_dummies(dataset['Geography'],drop_first = True)

df1 = pd.get_dummies(dataset['Gender'],drop_first = True)

dataset = pd.concat([df, dataset], axis = 1)
dataset = pd.concat([df1, dataset], axis = 1)

#Dropping original variables(Objects)
dataset.drop('Geography', axis = 1, inplace = True)
dataset.drop('Gender', axis = 1, inplace = True)

#Dropping Unwanted information
dataset.drop('RowNumber', axis = 1, inplace = True)
dataset.drop('CustomerId', axis = 1, inplace = True)
dataset.drop('Surname', axis = 1, inplace = True)


#Taking X and Y from Datasets
x = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

#Splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#Feature Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Evaluating The model using K Cross Fold 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model():
    
    #Initilizing the ANN
    model = Sequential()
    
    #Adding first Layer and Hidden Layer
    model.add(Dense(units = 6, input_dim =11 ,kernel_initializer='glorot_uniform', activation = 'relu'))
    
    #Adding second Hidden Layer
    model.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu'))
    
    #Adding output layer
    model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))
    
    #Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    
    return model

classifier = KerasClassifier(build_fn= build_model, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
    
mean = accuracies.mean()
sd = accuracies.std()   
