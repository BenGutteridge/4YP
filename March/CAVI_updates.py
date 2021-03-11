# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:22:10 2021

@author: benpg

CAVI update equations for the **known-covariance** case
"""

import numpy as np
from numpy.linalg import inv

def alpha_k(Nk, alpha0):
  return alpha0 + Nk

def m_invC_k(Nk, xkbar, m0, invC0, invSig):
    """
    Parameters
    ----------
    Nk : float
        Sum of responsibilities rnk for Gaussian k.
    xkbar : 2-vector
        Sum of dataset points weighted by responsibility of Gaussian k, averaged over Nk.
    m0 : 2-vector
        Prior for mean of Gaussian-distributed mu_k.
    invC0 : 2x2 array
        Prior for inverse covariance of Gaussian-distributed mu_k with mean m_k.
    invSig : 2x2 array
        Constant, known inverse covariance of Gaussian-disribution point in cluster k with mean mu_k.

    Returns
    -------
    m_k : 2-vector
        Updated variational parameter for mean of distribution of mu_k.
    invC_k : TYPE
        Updated variational parameter for inverse cov of distribution of mu_k.

    """
    if Nk==0:
        return m0, invC0 # component is dead, weight -> 0
    else:
        invC_k = invC0 + invSig*Nk
        U = np.dot(m0, invC0) + np.dot(Nk*xkbar, invSig) 
        m_k =  np.dot(U, inv(invC_k)).reshape(2,)
    return m_k, invC_k
