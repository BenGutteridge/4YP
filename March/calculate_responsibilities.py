# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:52:55 2021

@author: benpg
"""

import numpy as np
from numpy.linalg import inv, det, multi_dot
from scipy.special import digamma, gammaln
import sys

def E_ln_pi_k(k, alpha):
    return digamma(alpha[k]) - digamma(np.sum(alpha))

def E_N_exp_k(k, m, C, invSig, xn):
    # using modified Bishop eqn (10.64) and Matrix Cookbook eq (380)
    v = m[k] - xn # 1x2 row vector
    return multi_dot((v, invSig, v.T)) + np.trace(np.dot(invSig, C[k]))

def ln_rho_nk(k, alpha, m, C, invSig, xn, D=2):
    Elnpik = E_ln_pi_k(k,alpha)
    ## THIS BIT BELOW
    ENexpk = E_N_exp_k(k, m, C, invSig, xn)
    return Elnpik + 0.5*det(invSig) - 0.5*D*np.log(2*np.pi) - 0.5*ENexpk


def r_nk(k, alpha, m, C, invSig, xn):
    rhonk = np.exp(ln_rho_nk(k, alpha, m, C, invSig, xn, D=2))
    sum_k_rho = 0.
    for j in range(alpha.shape[0]):
        ln_rhonk = ln_rho_nk(j, alpha, m, C, invSig, xn, D=2)
        if np.isnan(ln_rhonk): 
            print('\nln_rho_nk() returning nan')
            print(alpha, m , C, xn)
            sys.exit()
        sum_k_rho = sum_k_rho + np.exp(ln_rhonk)
    if sum_k_rho>0: 
        return rhonk/sum_k_rho
    else: return 0.