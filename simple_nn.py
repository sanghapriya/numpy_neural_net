#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:20:46 2017

@author: sangho
"""

import numpy as np
import pandas as pd


def softmax(x):
    x -= np.max(x)
    sm = (np.exp(x).T / np.sum(np.exp(x),axis=1)).T
    return sm



def retrieveData():
    dataset = pd.read_csv("train.csv")
    
    Y = dataset['label']
    
    Y = Y.reshape(Y.shape[0],1)[:10000,:]
     
    X = dataset.values[:10000,1:]
    
    X= X.T/255
    
    return X,Y

def initializeWnB(fl_dim,layer_array):
    
    parameters = {}
    
    parameters["W1"] = np.random.randn(layer_array[1],fl_dim)*0.01
    
    parameters["b1"] = np.zeros(layer_array[1]).reshape(layer_array[1],1)
    
    for l in range(len(layer_array)):
        
        parameters["W"+str(l+1)] = np.random.randn(layer_array[l],layer_array[l-1])*0.01
    
        parameters["b"+str(l+1)] = np.zeros(layer_array[1]).reshape(layer_array[1],1)
    
    return parameters

        
print(initializeWnB(100,[4,5,10]))

def predict():
    
    X,Y,W,b = initialize()


    Y_h = softmax(np.dot(W.T,X)+b)

    print(Y_h.shape)
