# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:40:38 2021

@author: benpg

Calculating ELBO for KNOWN covariance case
"""

# should probably do elbo from scratch

import numpy as np
from numpy.linalg import inv, det, multi_dot
from scipy.special import gammaln
from calculate_responsibilities import E_N_exp_k, E_ln_pi_k

def E_ln_p_X_given_Z_mu(m, C, invSig, C0, NK, xbar, SK, D=2):
    # needs checking - two Traces?
    Ksum = 0.0
    for k in range(len(m)):
        a = np.log(det(invSig))
        b = -D/2 * np.log(2*np.pi)
        c = -np.trace(np.dot(invSig, C0))
        d = -np.trace(np.dot(invSig, SK[k]))
        v = (m[k] - xbar[k]).reshape(2,1)
        e = -multi_dot((v.T, invSig, v))
        Ksum += 0.5*NK[k]*(a + b + c + d + e)
    return float(Ksum)    
    

def E_ln_p_Z_given_pi(r, alpha):
  sum = 0.
  N = len(r[0])
  for n in range(N):
    for k in range(alpha.shape[0]):
      sum = sum + r[k][n]*E_ln_pi_k(k,alpha)
  return sum

def ln_C(alpha):
  return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

def ln_B(W,nu,D=2):
  ln_num = -0.5*nu*np.log(det(W))
  ln_det_1 = 0.5*nu*D*np.log(2) - D*(D-1)*0.25*np.log(np.pi)
  ln_det_2 = np.sum(gammaln(np.array([0.5*(nu+1-i) for i in range(D)])))
  return  ln_num - ln_det_1 - ln_det_2

def E_ln_p_pi(alpha0, alpha):
    K=alpha.shape[0]
    return ln_C(alpha0*np.ones(K)) + (alpha0-1)*np.sum(np.array([E_ln_pi_k(k,alpha) for k in range(K)]))


def E_ln_p_mu(m, C, m0, invC0, D=2):
    a = -D*np.ln(np.pi*2)
    b = np.log(det(invC0))
    v = (m-m0).reshape(2,1)
    c = -multi_dot(v.T, invC0, v) - np.trace(np.dot(invC0, C))
    return 0.5*(a+b+c)

def E_ln_q_Z(r):
    sum = 0.
    N, K = len(r[0]), len(r) # r is a list of K lists of length N
    for n in range(N):
        for k in range(K):
            if r[k][n] > 0: # n.b. returns nan for 0log0, so need to bypass
                sum = sum + r[k][n]*np.log(r[k][n])
    return sum

def E_ln_q_pi(alpha):
  K = alpha.shape[0]
  return np.sum(np.array([(alpha[k]-1)*E_ln_pi_k(k,alpha) for k in range(K)])) + ln_C(alpha)

def H_q_lam_k(k,nu,W,D=2):
  return -ln_B(W[k],nu[k]) - 0.5*(nu[k]-D-1)*E_ln_lam_k(k, nu, W) + 0.5*nu[k]*D

def E_ln_q_mu_lam(beta,W,nu,D=2):
  sum = 0.
  for k in range(beta.shape[0]):
    sum=sum+0.5*E_ln_lam_k(k,nu,W)+0.5*D*np.log(beta[k]/(2*np.pi))-0.5*D-H_q_lam_k(k,nu,W)
  return sum

def calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,beta0,m0,W0,nu0):
  p1 = E_ln_p_X_given_Z_mu_lam(beta, m, W, nu, NK, S, xbar)
  p2 = E_ln_p_Z_given_pi(r, alpha)
  p3 = E_ln_p_pi(alpha0, alpha)
  p4 = E_ln_p_mu_lam(beta, m, W, nu, beta0, m0, W0, nu0) # this one is an array
  q1 = E_ln_q_Z(r)
  q2 = E_ln_q_pi(alpha)
  q3 = E_ln_q_mu_lam(beta,W,nu)
  # print(type(p1),type(p2),type(p3),type(p4),type(q1),type(q2),type(q3))
  # print(p1)
  # if np.sum(np.isnan([p1,p2,p3,p4,q1,q2,q3]))>0:
  #   print('\nArgs of calculate ELBO:\nr,alpha,beta,m,W,nu,NK,S,xbar,alpha0,m0,W0,nu0')
  #   print(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,m0,W0,nu0)
  # assert not np.isnan(p1)
  # assert not np.isnan(p2)
  # assert not np.isnan(p3)
  # assert not np.isnan(p4)
  # assert not np.isnan(q1)
  # assert not np.isnan(q2)
  # assert not np.isnan(q3)
    
  return p1+p2+p3+p4-q1-q2-q3