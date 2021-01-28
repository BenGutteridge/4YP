# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:38:11 2021

@author: benpg
"""

import numpy as np
from numpy.linalg import inv, det, multi_dot
import scipy
from scipy.special import digamma
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

# Variational priors
alpha0 = 0.1
beta0 = 0.1
m0 = np.zeros(2)
W0 = np.eye(2)
nu0 = 2

# Set priors, dataset, problem
D = 2
K = 5
N = 500

# Dataset
centres = [np.array([0.,8.]), np.array([5.,0.])]
covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])] 
X1 = multivariate_normal(mean=centres[0],
                         cov=covs[0],
                         size=int(N/2))
X2 = multivariate_normal(mean=centres[1],
                         cov=covs[1],
                         size=int(N/2))
X = np.concatenate((X1,X2))

def alpha_k(Nk, alpha0=alpha0):
  return alpha0 + Nk

def beta_k(Nk, beta0=beta0):
  return beta0 + Nk

def m_k(Nk, xkbar, betak, m0=m0, beta0=beta0):
  return (1/betak)*(beta0*m0 + Nk*xkbar)

def W_k(Nk, xkbar, Sk, m0=m0, beta0=beta0):
  inv_Wk = inv(W0) + Nk*Sk + ((beta0*Nk)/(beta0+Nk))*np.dot((xkbar-m0),(xkbar-m0).T) 
  return inv(inv_Wk)

def nu_k(Nk, nu0=nu0):
  return nu0 + Nk

def N_k(responsibilities_k):
  return np.sum(responsibilities_k)

def x_k_bar(Nk, responsibilities_k, X):
  N = responsibilities_k.shape[0]
  D = X.shape[1]
  sum = np.zeros((1,D))
  for n in range(N):
    sum = sum + responsibilities_k[n]*X[n]
  return (1/Nk)*sum

def S_k(Nk, responsibilities_k, X, xkbar):
  N = responsibilities_k.shape[0]
  sum = 0.0
  for n in range(N):
    sum = sum + responsibilities_k[n]*np.dot((X[n]-xkbar).T,(X[n]-xkbar))
    
  return (1/Nk)*sum

def ln_rho_nk(k, alpha, nu, W, beta, m, xn):
  D = X.shape[1] # dimensionality
  
  E_ln_pi_k =  digamma(alpha[k]) - digamma(np.sum(alpha))
  E_ln_lam_k = np.sum(digamma(nu[k]+1-np.arange(D)+1)) + D*np.log(2) + np.log(det(W[k]))
  E_ln_mu_k = D*beta[k]**-1 + nu[k]*multi_dot(((xn-m[k]).T, W[k], (xn-m[k])))
  
  return E_ln_pi_k + 0.5*E_ln_lam_k - 0.5*D*np.log(2*np.pi) - 0.5*E_ln_mu_k

def r_nk(k, alpha, nu, W, beta, m, xn):
  rhonk = np.exp(ln_rho_nk(k, alpha, nu, W, beta, m, xn))
  sum_k_rho = 0.
  for j in range(K):
    sum_k_rho += np.exp(ln_rho_nk(j, alpha, nu, W, beta, m, xn))
  return rhonk/sum_k_rho

def M_step(r, alpha,nu,W,beta,m,X):
  for k in range(K):
    Nk = N_k(r[:,k])
    print('N_%d = '%k, Nk)
    xkbar = x_k_bar(Nk, r[:,k], X)
    Sk = S_k(Nk, r[:,k], X, xkbar)

    alpha[k] = alpha_k(Nk)
    beta[k] = beta_k(Nk)
    m[k] = m_k(Nk, xkbar, beta[k])
    W[k] = W_k(Nk, xkbar, Sk)
    nu[k] = nu_k(Nk)

    # print('k=%d\nalpha'%k, alpha[k], '\nbeta', beta[k], '\nm', m[k], '\nW', W[k], '\nnu', nu[k])
  return alpha, beta, m, W, nu

# Run the code

alpha = 0.1*np.ones(K)
beta = 0.1*np.ones(K)
m = np.zeros((K,D))
W = [np.eye(D) for _ in range(K)]
nu = 2*np.ones(K)

r = np.empty((N,K))
for n in range(N):
    for k in range(K):
        r[n,k] = r_nk(k, alpha, nu, W, beta, m, X[n])

alpha, beta, m, W, nu = M_step(r, alpha,nu,W,beta,m,X)
