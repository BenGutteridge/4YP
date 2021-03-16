# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:10:53 2021

@author: benpg

Calculating responsibilities for unknown covariance
"""
import numpy as np
from numpy.linalg import inv, det, multi_dot
from scipy.special import digamma, gammaln
import sys

def E_ln_pi_k(k, alpha):
  return digamma(alpha[k]) - digamma(np.sum(alpha))

def E_ln_lam_k(k, nu, W, D=2):
  return np.sum(digamma(nu[k]+1-np.arange(D)+1)) + D*np.log(2) + np.log(det(W[k]))

def E_ln_mu_k(k, beta, m, W, nu, xn, D=2):
  # print('m, W\n', m, W)
  return D*beta[k]**-1 + nu[k]*multi_dot(((xn-m[k]), W[k], (xn-m[k]).T))

def ln_rho_nk(k, alpha, nu, W, beta, m, xn, D=2):
  return E_ln_pi_k(k,alpha) + 0.5*E_ln_lam_k(k, nu, W) - 0.5*D*np.log(2*np.pi) - 0.5*E_ln_mu_k(k, beta, m, W, nu, xn)


def r_nk(k, alpha, nu, W, beta, m, xn):
  rhonk = np.exp(ln_rho_nk(k, alpha, nu, W, beta, m, xn))
  sum_k_rho = 0.
  for j in range(alpha.shape[0]):
      ln_rhonk = ln_rho_nk(j, alpha, nu, W, beta, m, xn)
      if np.isnan(ln_rhonk): 
          print('\nln_rho_nk() returning nan')
          print(alpha, beta, m , W, nu, xn)
          sys.exit()
      sum_k_rho = sum_k_rho + np.exp(ln_rhonk)
  if sum_k_rho>0: 
     return rhonk/sum_k_rho
  else: return 0.