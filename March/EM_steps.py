# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:20:42 2021

@author: benpg
"""


import numpy as np
from numpy.linalg import inv
from statistics_of_observed_data import N_k, x_k_bar, S_k
from CAVI_updates import alpha_k, m_invC_k
from calculate_responsibilities import r_nk


# E step: calculate responsibility
def E_step(N, K, alpha, m, C, invSig, X):
  r = []
  for k in range(K):
    r.append([])
    for n in range(N):
      r[k].append(r_nk(k, alpha, m, C, invSig, X[n]))
  return r


# M step: update hyperparameters
def M_step(r,X,alpha0,m0,invC0,invSig,K):
  NK, xbar, SK = [],[],[]
  alpha = np.empty(K)
  m, invC = [np.zeros(2) for _ in range(K)], [np.zeros((2,2)) for _ in range(K)]

  for k in range(K):
    Nk = N_k(r[k])
    xkbar = x_k_bar(Nk, r[k], X)
    Sk = S_k(Nk, r[k], X, xkbar)

    alpha[k] = alpha_k(Nk, alpha0)
    m[k], invC[k] = m_invC_k(Nk, xkbar, m0, invC0, invSig)

    NK.append(Nk)
    xbar.append(xkbar.reshape(2,))
    SK.append(Sk)

    # print('k=%d\nalpha'%k, alpha[k], '\nbeta', beta[k], '\nm', m[k], '\nW', W[k], '\nnu', nu[k])
  return alpha, m, inv(invC), NK, xbar, SK