# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:04:27 2021

@author: benpg

Statistics of observed data
"""

import numpy as np

def N_k(responsibilities_k):
  return np.sum(np.array(responsibilities_k))

def x_k_bar(Nk, responsibilities_k, X, D=2):
  sum = np.zeros((1,D))
  for n in range(X.shape[0]):
    sum = sum + responsibilities_k[n]*X[n]
  if Nk > 0:
    return (1/Nk)*sum
  else:
    return np.zeros(2) # component is dead

def S_k(Nk, responsibilities_k, X, xkbar):
  sum = 0.0
  for n in range(X.shape[0]):
    sum = sum + responsibilities_k[n]*np.dot((X[n]-xkbar).T,(X[n]-xkbar))
  if Nk > 0:
    return (1/Nk)*sum
  else:
    return np.eye(2) # doesn't actually matter what it returns, the component is dead