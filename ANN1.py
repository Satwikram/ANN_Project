# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:39:54 2020

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

#Importing keras packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#Initilizing the ANN
model = Sequential()

#Adding first Layer and Hidden Layer
model.add(Dense(units = 6, input_dim =11 ,kernel_initializer='glorot_uniform', activation = 'relu'))
model.add(Dropout(rate = 0.1))

#Adding second Hidden Layer
model.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu'))
model.add(Dropout(rate = 0.1))

#Adding output layer
model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))

#Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Fitting ANN with Training sets
model.fit(x_train, y_train, batch_size = 10, epochs = 100)

#Predicting the test set
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)


'''
Single prediction details
Geography : France
Credit Score : 600
Gender : Male
Age : 40
Tennure : 3
Balance : 60,000
No of Products : 2
Has Credit Card : Yes
Is Active Member : Yes
Estimated Salary is : 50,000
Predict Whether the Customer will leave the bank or not
'''


#Single Prediction
# Enter data in order 
new_pred = model.predict(sc.transform(np.array([[1, 0, 0, 600, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)


#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)
