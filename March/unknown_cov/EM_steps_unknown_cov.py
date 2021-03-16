# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:16:23 2021

@author: benpg

EM steps for unknown covariance CAVI
"""

import numpy as np
from statistics_of_observed_data import N_k, x_k_bar, S_k
from CAVI_updates_unknown_cov import alpha_k, beta_k, m_k, W_k, nu_k
from calculate_responsibilities_unknown_cov import r_nk
from phi_grad_funcs_unknown_cov import *


# E step: calculate responsibility
def E_step(N,K,alpha,nu,W,beta,m,X):
  r = []
  for k in range(K):
    r.append([])
    for n in range(N):
      r[k].append(r_nk(k, alpha, nu, W, beta, m, X[n]))
  return r


# M step: update hyperparameters
def M_step(r,X,alpha0,beta0,m0,W0,nu0,K):
  NK, xbar, S = [],[],[]
  alpha, beta, nu = np.empty(K), np.empty(K), np.empty(K)
  m, W = [np.zeros(2) for _ in range(K)], [np.zeros((2,2)) for _ in range(K)]

  for k in range(K):
    Nk = N_k(r[k])
    xkbar = x_k_bar(Nk, r[k], X)
    Sk = S_k(Nk, r[k], X, xkbar)

    alpha[k] = alpha_k(Nk, alpha0)
    beta[k] = beta_k(Nk, beta0)
    m[k] = m_k(Nk, xkbar, beta[k], m0, beta0)
    W[k] = W_k(Nk, xkbar, Sk, m0, beta0, W0)
    nu[k] = nu_k(Nk, nu0)

    NK.append(Nk)
    xbar.append(xkbar)
    S.append(Sk)

    # print('k=%d\nalpha'%k, alpha[k], '\nbeta', beta[k], '\nm', m[k], '\nW', W[k], '\nnu', nu[k])
  return alpha, beta, m, W, nu, NK, xbar, S


# Alternative M-step, using gradient ascent
def M_step_GD(X, r, alpha, beta, m, W, nu, alpha0, beta0, m0, W0, nu0, K, D, step=0.1):
    NK, xbar, S = [],[],[]
    for k in range(K):
      NK.append(N_k(r[k]))
      xbar.append(x_k_bar(NK[k], r[k], X))
      S.append(S_k(NK[k], r[k], X, xbar[k]))
      
    d_alpha = Dalpha(NK, alpha, alpha0, K)
    # d_beta = Dbeta(NK, beta, beta0, D, K)
    d_m = Dm(NK, xbar, m, W, nu, beta0, m0, K)
    # d_W = DW(NK, xbar, S, W, nu, beta0, m0, W0, nu0, K)
    # d_nu = Dnu(NK, xbar, S, W, nu, beta0, m0, W0, nu0, K, D)
    # print('Gradients:\n',d_alpha, d_beta, d_m, d_W)
    # print(alpha, beta, m, W)
      
    step_alpha = 1. 
    alpha += step_alpha*d_alpha
    # beta += step*d_beta
    # nu += step*d_nu
    for k in range(K):
      m[k] = m[k] + step*d_m[k]
      # W[k] = W[k] + step*d_W[k]
      
    return alpha, beta, m, W, nu, NK, xbar, S