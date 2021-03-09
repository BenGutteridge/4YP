# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:05:55 2021

@author: benpg

Gradient functions for GMM
"""
from scipy.special import polygamma as psi_prime
import numpy as np
from numpy.linalg import inv
import autograd.numpy as anp
from autograd.scipy.special import digamma
from autograd import grad


def Dalpha(Nk, alpha, alpha0, K):
    d_alpha = np.zeros(K)
    alpha_hat = np.sum(alpha)
    for k in range(K):
        d_alpha[k] = (Nk[k]+alpha0-alpha[k])*(psi_prime(1,alpha[k]) - psi_prime(1,alpha_hat))
    return d_alpha

def Dbeta(Nk, beta, beta0, D, K):
    d_beta = np.zeros(K)
    for k in range(K):
        d_beta[k] = (D/2)*(beta[k])**(-2)*(Nk[k]+beta0-beta[k])
    return d_beta

def Dm(Nk, xkbar, m, W, nu, beta0, m0, K):
    d_m = []
    for k in range(K):
        d_m_k = np.dot(nu[k]*W[k], (Nk[k]*(xkbar[k]-m[k]) - beta0*(m[k]-m0)).T)
        d_m.append(d_m_k.reshape(2,))
    return d_m

# cavi update for W, needed for calculating gradient of ELBO wrt Wk
def W_k(Nk, xkbar, Sk, m0, beta0, W0):
  inv_Wk = inv(W0) + Nk*Sk + ((beta0*Nk)/(beta0+Nk))*np.dot((xkbar-m0).T,(xkbar-m0)) 
  return inv(inv_Wk)

def DW(Nk, xkbar, Sk, W, nu, beta0, m0, W0, nu0, K):
    d_W = []
    for k in range(K):
        W_hat_k = W_k(Nk[k], xkbar[k], Sk[k], m0, beta0, W0)
        d_W.append(0.5*(Nk[k]+nu0+nu[k])*inv(W[k]) - 0.5*nu[k]*(inv(W_hat_k)-inv(W[k])))
    return d_W

def ln_lam_tilde_k(k, nu, W, D):
  return anp.sum(digamma(nu[k]+1-anp.arange(D)+1)) + D*anp.log(2) + anp.log(anp.det(W[k]))

def Dnu(Nk, xkbar, Sk, W, nu, beta0, m0, W0, nu0, K, D):
    d_nu = np.zeros(K)
    d_ln_lam_d_nu = grad(ln_lam_tilde_k, 1)
    for k in range(K):
        W_hat_k = W_k(Nk[k], xkbar[k], Sk[k], m0, beta0, W0)
        d_nu[k] = 0.5*((Nk[k]+nu0-nu[k])*d_ln_lam_d_nu(k,nu,W,D) - np.trace(np.dot(W_hat_k,W[k])) + D)
    return d_nu

