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

def relu(x):
    return max(0,x),x

def sigmoid(x):
    return (1/(1+np.exp(-1*x))),x

def derv_relu(x):
    return relu(x)/x

def derv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def retrieveData():    
    dataset = pd.read_csv("train.csv")    
    Y = dataset['label']    
    Y = Y.reshape(Y.shape[0],1)[:10000,:]     
    X = dataset.values[:10000,1:]    
    X= X.T/255    
    return X,Y


def initialize_w_b(layer_array):    
    parameters = {}
    for l in range(1,len(layer_array)):        
        parameters["W"+str(l)] = np.random.randn(layer_array[l],layer_array[l-1])*0.01    
        parameters["b"+str(l)] = np.zeros(layer_array[1]).reshape(layer_array[1],1)    
    return parameters


def forward_prop(A,W,b):    
    Z = np.dot(W,A)+b    
    cache = (W,A,b)    
    return Z,cache

def back_prop(dz,cache):
    W,A,b,A_prev = cache
    m= A_prev.shape[1] 
    dW = (1/m)*np.dot(dz,A_prev.T)
    db = (1/m)*np.sum(dz,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dz) 
    return dA_prev, dW, db


def linear_forward_prop(A_prev,W,B,activation):
    if activation =="sigmoid":
        Z,linear_cache = forward_prop(A_prev,W,B)
        A, activation_cache = sigmoid(Z)  
    if activation == "relu":
        Z,linear_cache = forward_prop(A_prev,W,B)
        A, activation_cache = relu(Z)  
    cache = (linear_cache,activation_cache)
    return A,cache

def linear_backward_prop(da,cache,activation):
    linear_cache,activation_cache = cache
    
    if activation == "sigmoid":
        dz = back_prop(da,linear)
    
def L_model_forward(X, parameters):
    
    caches = []
    A=X
    L = len(parameters)//2
    
    for l in range(1,L):
        A_prev = A        
        A,cache = linear_forward_prop(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
    
    AL,cache = linear_forward_prop(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    caches.append(cache)
    
    return caches,AL


def L_model_backward()

def compute_cost(AL,Y):
    
    m = Y.shape[1]
    
    cost = (-1/m)*(np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL),axis=1))
    
    cost = np.squeeze(cost)
    
    return cost



def predict():
    
    X,Y,W,b = initialize()


    Y_h = softmax(np.dot(W.T,X)+b)

    print(Y_h.shape)
