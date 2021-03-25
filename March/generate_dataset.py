# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:28:46 2021

@author: benpg

Generate 2D dataset
"""
import numpy as np
from numpy.random import multivariate_normal

# Same dataset each time
np.random.seed(42)

def generate_2D_dataset(N, K=2, centres=None, covs=None, weights=None):
    """

    Parameters
    ----------
    N : int
        Number of 2D points wanted.
    K : int, optional
        Number of clusters of points. The default is 2.
    centres : list of 2-vector numpy arrays, optional
        Means of GMM generating data. The default is None.
    covs : list of 2x2 matrix numpy arrays, optional
        Covariances of GMM generating data. The default is None.
    weights : 1D numpy array, optional
        Mixture weights of GMM generating data. The default is None.

    Returns
    -------
    X : Nx2 numpy array
        Dataset, x-y coords.
    centres : list of 2-vector numpy arrays, optional
        Means of GMM generating data. The default is None.
    covs : list of 2x2 matrix numpy arrays, optional
        Covariances of GMM generating data. The default is None.

    """
    # default
    if centres==None or covs==None:
        centres = [np.array([0.,8.]), np.array([5.,0.])]
        covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])] 
        if K>2:
            for k in range(2,K):
                centres.append(np.random.randint(10, size=2))
                covs.append(np.eye(2))
    if weights is None:
        weights = np.ones(K)/K
            
    # generate dataset
    N_per_cluster = np.around(weights*N)
        
    X = multivariate_normal(mean=centres[0],
                             cov=covs[0],
                             size=int(N_per_cluster[0]))
    for k in range(1,K):
        X = np.concatenate((X,
                            multivariate_normal(mean=centres[k],
                                 cov=covs[k],
                                 size=int(N_per_cluster[k]))))
    return X, centres, covs, weights