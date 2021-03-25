# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:52:55 2021

@author: benpg
"""

import numpy as np
from numpy.linalg import det, multi_dot
from scipy.special import digamma
import sys

def E_ln_pi_k(k, alpha):
    """Bishop Eqn B.21"""
    return digamma(alpha[k]) - digamma(np.sum(alpha))

def E_N_exp_k(k, m, C, invSig, xn):
    """Using modified Bishop eqn (10.64) and Matrix Cookbook eq (380)"""
    v = m[k] - xn # 1x2 row vector
    return multi_dot((v, invSig, v.T)) + np.trace(np.dot(invSig, C[k]))

def calculate_ln_rho_nk(k, alpha, m, C, invSig, xn, D=2):
    """
    Adapted from Bishop Eqn 10.46
    Also Xie Eqs 87/88, 100
    """
    Elnpik = E_ln_pi_k(k,alpha)
    ENexpk = E_N_exp_k(k, m, C, invSig, xn)
    return Elnpik + 0.5*det(invSig) - 0.5*D*np.log(2*np.pi) - 0.5*ENexpk


def r_nk(k, alpha, m, C, invSig, xn):
    """
    Calculates responsibility borne by each mixture component for each datapoint xn
    Adapted from Bishop Eqn 10.49
    """
    try:
        rho_nk = np.exp(calculate_ln_rho_nk(k, alpha, m, C, invSig, xn, D=2))
        Ksum_rho_n = 0.
        for j in range(alpha.shape[0]):
            ln_rho_nk = calculate_ln_rho_nk(j, alpha, m, C, invSig, xn, D=2)
            assert not np.isnan(ln_rho_nk)
            Ksum_rho_n = Ksum_rho_n + np.exp(ln_rho_nk)
        r_nk = float(rho_nk)/float(Ksum_rho_n)
        assert not np.isnan(r_nk)
        return r_nk
        
    except AssertionError: 
        # NaN error, probably due to exp overflow or ln(0)
        print('\nln_rho_nk() or r_nk returning nan\n')
        print(alpha, m , C, xn)
        sys.exit()
    except ZeroDivisionError: 
        # When all K components bear negligible responsibility
        return 0.