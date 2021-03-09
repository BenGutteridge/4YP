# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:01:18 2021

@author: benpg

CAVI update equations
"""

import numpy as np
from numpy.linalg import inv

def alpha_k(Nk, alpha0):
  return alpha0 + Nk

def beta_k(Nk, beta0):
  return beta0 + Nk

def m_k(Nk, xkbar, betak, m0, beta0):
  return ((1/betak)*(beta0*m0 + Nk*xkbar)).reshape(2)

def W_k(Nk, xkbar, Sk, m0, beta0, W0):
  inv_Wk = inv(W0) + Nk*Sk + ((beta0*Nk)/(beta0+Nk))*np.dot((xkbar-m0).T,(xkbar-m0)) 
  return inv(inv_Wk)

def nu_k(Nk, nu0):
  return nu0 + Nk