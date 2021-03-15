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
from grad_funcs import L_grad_alpha, L_grad_m, L_grad_invC


# E step: calculate responsibility
def E_step(N, K, alpha, m, C, invSig, X, alpha_lb=0.1):
    r = []
    for k in range(K):
        if alpha[k] <= alpha_lb: 
            r.append(np.zeros(N).tolist()) # should probably fix
        else:
            r.append([])
            for n in range(N):
                r[k].append(r_nk(k, alpha, m, C, invSig, X[n]))
    return r


# M step: update hyperparameters
def M_step(r,X, alpha0,m0,invC0,invSig,K):
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

def M_step_GD(r, X, alpha, m, C, alpha0, m0, invC0, invSig, K,
              step_sizes={'alpha': 1.0, 'm': 1e-4, 'C': 1e-4}):
    # performing GD update for M step rather than CAVI update equations
    
    invC = inv(C)
    
    NK = [N_k(r[k]) for k in range(K)]
    xbar = [x_k_bar(NK[k], r[k], X) for k in range(K)]
    SK = [S_k(NK[k], r[k], X, xbar[k]) for k in range(K)]
    
    # Gradient update equations
    d_alpha = L_grad_alpha(alpha0, alpha, NK)
    d_m = L_grad_m(m, m0, invC0, invSig, NK, xbar)
    d_invC = L_grad_invC(m, invC, invSig, invC0, NK)

    for k in range(K): 
        # constraints: alpha>0. Setting alpha>.1 as psi'(alpha->0) -> inf
        alpha[k] = np.max((0.1, alpha[k] + d_alpha[k]*step_sizes['alpha']))
        m[k] = m[k] + d_m[k]*step_sizes['m']
        invC[k] = invC[k] + d_invC[k]*step_sizes['invC']
        
    return alpha, m, inv(invC), NK, xbar, SK


def perturb_variational_params(alpha=None, m=None, C=None):
    if alpha is not None:
        K = alpha.shape[0]
        alpha = np.ones(K)*(np.sum(alpha)/K)
    if m is not None:
        m = [m[k] + np.random.randint(10,20,size=2)-10 for k in range(len(m))]
    if C is not None:
        C = [np.eye(2) for _ in range(len(C))]
    return alpha, m, C 
    