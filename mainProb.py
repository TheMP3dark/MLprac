# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:11:16 2019

@author: TheMP3dark
"""

#run this program on your local pyhthon interpreter, provided you have installed the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Function importing dataset
def importdata():
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',sep=',',header=None)
    
    #Printing the dataset shape
    print("Dataset  length:",len(balance_data))
    print("Dataset shape:",balance_data.shape)
    
    #printing the dataset observations
    print ("Dataset: ",balance_data.head())
    return balance_data

#Function to split the dataset
def splitdataset(balance_data):
    
    #Separating the target variable
    X = balance_data.values[:,1:5]
    Y = balance_data.values[:,0]
    
    #Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    
    return X, Y, X_train, X_test, y_train, y_test


def train_using_gini(X_train, X_test, y_train):
    #creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf=5)
    
    #Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

#function to perform training with entropy
def train_using_entropy(X_train, X_test, y_train):
    
    #decision tree with entropy
    clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state = 100, max_depth = 3, min_samples_leaf = 5)
    
    #performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


#Function to make prediction
def prediction(X_test, clf_object):
    #prediction on test with giniInderx
    y_pred = clf_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred
        
#Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
    
    print("Accuracy: ",accuracy_score(y_test,y_pred)*100)
    
    print("Report: ",classification_report(y_test, y_pred))

#Driver code
def main():
    #building phase
    data = importdata()
    X,Y,X_train,X_test,y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train,X_test,y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    
    #Operational Phase
    print("Results using gini index: ")
    
    #Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    
    print("Results Using Entropy: ")
    #Prediction using entropy
    y_pred_entropy = prediction (X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
    
#calling main function 
    if __name__=="__main__":
        main()
