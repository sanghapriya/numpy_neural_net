#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 08:14:37 2017

@author: sangho
"""

import numpy as np

def softmax(x):    
    #print("Softmax initiated")
    x -= np.max(x)
    sm = (np.exp(x) / np.sum(np.exp(x),axis=0))
    return sm

def relu(x):
    
    #print("Relu initiated")
    
    return np.maximum(x,0)

def sigmoid(x):
    #print("Sigmoid initiated")
    return (1/(1+np.exp(-1*x)))

def derv_relu(x):
    
    return 1. * (x > 0)

def derv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def derv_softmax(x):  

    return np.multiply( x, 1 - x ) + sum(
            - x * np.roll( x, i, axis = 1 )
            for i in range(1, x.shape[1] )
        )