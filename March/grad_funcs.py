# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:15:12 2021

@author: benpg
"""
from scipy.special import polygamma as psi_prime
import numpy as np
from numpy.linalg import inv, det, multi_dot
import autograd.numpy as anp
from autograd.scipy.special import digamma
from autograd import grad

def L_grad_alpha(alpha0, alpha, NK):
    K = alpha.shape[0]
    d_alpha = np.zeros(K)
    alpha_hat = np.sum(alpha)
    for k in range(K):
        a = (NK[k]+alpha0-alpha[k])
        b = (psi_prime(1,alpha[k]) - psi_prime(1,alpha_hat))
        c = digamma(alpha[k]) - digamma(alpha_hat)
        d_alpha[k] = (a * b) - c
    return d_alpha*(alpha>0.1)

def H_grad_alpha():
    pass

def L_grad_m(m, m0, invC0, invSig, NK, xbar):
    K, d_m = len(m), []
    for k in range(K):
        a = NK[k]*np.dot((m[k]-xbar[k]), invSig)
        b = np.dot((m[k]-m0), invC0)
        d_m.append((-a-b).reshape(2,))
    return d_m

# Based on Xie and exponential family stuff - needs another look
def L_grad_invC(m, invC, invSig, invC0, NK):
    K, d_C = len(invC), []
    for k in range(K):
        m[k].reshape(2,1)
        log_norm = 0.5*multi_dot((m[k].T, invC[k], m[k])) - 0.5*np.log(det(invC[k]))
        natural_grad = 0.25*(invC0 + NK[k]*invSig - invC[k]) 
        d_C.append(log_norm*natural_grad)
    return d_C

def L_grad_C(invC, invSig, invC0, NK):
    D, K = invSig.shape[0], len(invC)
    d_C = np.zeros((K,D,D))
    for k in range(K):
        d_C[k] = 0.5*(invC[k] - NK[k]*invSig - invC0)
    return d_C
    


# # cavi update for W, needed for calculating gradient of ELBO wrt Wk
# def W_k(Nk, xkbar, Sk, m0, beta0, W0):
#   inv_Wk = inv(W0) + Nk*Sk + ((beta0*Nk)/(beta0+Nk))*np.dot((xkbar-m0).T,(xkbar-m0)) 
#   return inv(inv_Wk)

# def DW(Nk, xkbar, Sk, W, nu, beta0, m0, W0, nu0, K):
#     d_W = []
#     for k in range(K):
#         W_hat_k = W_k(Nk[k], xkbar[k], Sk[k], m0, beta0, W0)
#         d_W.append(0.5*(Nk[k]+nu0+nu[k])*inv(W[k]) - 0.5*nu[k]*(inv(W_hat_k)-inv(W[k])))
#     return d_W

# def ln_lam_tilde_k(k, nu, W, D):
#   return anp.sum(digamma(nu[k]+1-anp.arange(D)+1)) + D*anp.log(2) + anp.log(anp.det(W[k]))

# def Dnu(Nk, xkbar, Sk, W, nu, beta0, m0, W0, nu0, K, D):
#     d_nu = np.zeros(K)
#     d_ln_lam_d_nu = grad(ln_lam_tilde_k, 1)
#     for k in range(K):
#         W_hat_k = W_k(Nk[k], xkbar[k], Sk[k], m0, beta0, W0)
#         d_nu[k] = 0.5*((Nk[k]+nu0-nu[k])*d_ln_lam_d_nu(k,nu,W,D) - np.trace(np.dot(W_hat_k,W[k])) + D)
#     return d_nu

# def D_rnk():
#     a = (ln_lam_tilde_k(k,nu,W,D) - D*beta[k]**(-1))*0.5
#     b = -np.log(rnk)-1 + 