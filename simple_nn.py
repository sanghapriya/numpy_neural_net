#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:20:46 2017

@author: sangho
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from activation_funcs import relu
from activation_funcs import softmax
from activation_funcs import derv_relu
from activation_funcs import derv_softmax



# Retrieves data from the csv file

def retrieve_data():    
    dataset = pd.read_csv("train.csv")    
    Y = dataset['label']    
    Y = Y.values.reshape(Y.shape[0],1)[:100,:]
    enc = OneHotEncoder(sparse=False)
    enc.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
    Y = enc.transform(Y).T
    
    X = dataset.values[:100,1:]    
    X= X.T/255    
    return X,Y

# Initialize the Weights and biases

def initialize_w_b(layer_array):    
    parameters = {}
    for l in range(1,len(layer_array)):        
        parameters["W"+str(l)] = np.random.randn(layer_array[l],layer_array[l-1])*0.01    
        parameters["b"+str(l)] = np.zeros(layer_array[l]).reshape(layer_array[l],1)       
    return parameters



# Perform Forward propagation

def L_model_forward(X, parameters):
    
    activations = {}
    A=X
    L = 4
    
    activations['A0']=X
    for l in range(1,L):
        A_prev = A        
        W =parameters['W'+str(l)]
        B= parameters['b'+str(l)]
        
        
        
        #print("iteration ",l)
        #print("A prev's shape",A_prev.shape)
        #print("W's shape",W.shape)
        #print("Dot product shape",np.dot(W,A_prev).shape)
        
        Z = np.dot(W,A_prev)+B
        if (l == L-1):
            A = softmax(Z)
        else:
            A = relu(Z)
        
        activations['A'+str(l)]=A

    AL = A
    
    return AL,activations


def L_model_backward(parameters,activations,Y):
    L = 4
   
    grad = {}
    
    for l in range(L-1,0,-1):
        m= activations['A'+str(l)].shape[1]
        if l == L-1:
            da_prev = - (np.divide(Y, activations['A'+str(l)]) - np.divide(1 - Y, 1 - activations['A'+str(l)]))
            #print("da prev shape",da_prev.shape)
            dz= np.multiply(da_prev,derv_softmax(activations['A'+str(l)]))
            
        else:   
            dz = np.multiply(da_prev,derv_relu(activations['A'+str(l)]))
        
        #print(" dz shape",dz.shape,"dw"+str(l))
        
        da_prev = np.dot(parameters['W'+str(l)].T,dz)
        grad['dw'+str(l)] = (1/m)*np.dot(dz,activations['A'+str(l-1)].T)
        grad['db'+str(l)] = (1/m)*np.sum(dz,axis=1,keepdims=True)
        
    #print("Grad :",grad)
        
    for l in range(L-1):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)]-0.1*grad['dw'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - 0.1*grad['db'+str(l+1)]
      
    return parameters
            
        
        
    



def compute_cost(AL,Y):
    
    m = Y.shape[1]
    
    cost = (-1/m)*np.sum((np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL),axis=1)))
    
    cost = np.squeeze(cost)
    
    return cost



def predict():
    
    X,Y = retrieve_data()
    #print(Y.shape)
    parameters = initialize_w_b([X.shape[0],50,40,Y.shape[0]])
    
    for i in range(100):

    
        AL,activations = L_model_forward(X,parameters)
        
        cost = compute_cost(AL,Y)
    
        print(cost)
        
        parameters = L_model_backward(parameters,activations,Y)
    
    
    
    

predict()
