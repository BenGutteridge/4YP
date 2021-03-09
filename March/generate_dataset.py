# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:28:46 2021

@author: benpg

Generate 2D dataset
"""
import numpy as np
from numpy.random import multivariate_normal


def generate_2D_dataset(N, K=2, centres=None, covs=None):
    # default
    if centres==None or covs==None:
        centres = [np.array([0.,8.]), np.array([5.,0.])]
        covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])] 
        if K>2:
            for k in range(2,K):
                centres.append(np.random.randint(10, size=2))
                covs.append(np.eye(2))
            
    
    # generate dataset
    N_per_k = np.floor_divide(N,K)
        
    X = multivariate_normal(mean=centres[0],
                             cov=covs[0],
                             size=int(N_per_k))
    for k in range(1,K):
        X = np.concatenate((X,
                            multivariate_normal(mean=centres[k],
                                 cov=covs[k],
                                 size=int(N_per_k))))
    return X, centres, covs