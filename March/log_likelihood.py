# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:40:01 2021

@author: benpg

Measuring goodness of fit
"""

import numpy as np
from scipy.stats import multivariate_normal as norm

def log_likelihood(X, pi, mu, cov):
    N = X.shape[0]
    # model is p(X|pi, mui) = PROD_n SUM_k pi_kN(xn|mu_k, Sigma)
    logsum = 0
    for n in range(N):
        logsum += ln_p_x_given_pi_mu(X[n], pi, mu, cov)
    return logsum
        
def ln_p_x_given_pi_mu(x, pi, mu, cov):
    K = pi.shape[0]
    lnp = pi * np.array([norm.pdf(x, mu[k], cov[k]) for k in range(K)])
    return np.log(np.sum(lnp))
        
    
