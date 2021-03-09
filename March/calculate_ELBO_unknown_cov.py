# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:25:48 2021

@author: benpg

Calulcating ELBO for unknown covariance
"""

import numpy as np
from numpy.linalg import inv, det
from scipy.special import gammaln
from calculate_responsibilities_unknown_cov import E_ln_lam_k, E_ln_mu_k, E_ln_pi_k

def E_ln_p_X_given_Z_mu_lam(beta, m, W, nu, NK, S, xbar, D=2):
  sum = 0.
  for k in range(beta.shape[0]):
    Eln_lam = E_ln_lam_k(k, nu, W)
    Tk = nu[k]*np.trace(np.dot(S[k],W[k]))
    Eln_mu = E_ln_mu_k(k,beta,m,W,nu,xbar[k])
    # print('\n%d\nEln_lam: '%k, Eln_lam)
    # print('Tl: ', Tk)
    # print('Eln_mu: ', Eln_mu)
    # print('NK: ', NK)
    sum = sum + NK[k]*(Eln_lam - Eln_mu - Tk - D*np.log(2*np.pi))
  #print(sum)
  #print(sum[0][0])
  return 0.5*sum

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

def E_ln_p_mu_lam(beta, m, W, nu, beta0, m0, W0, nu0, D=2):
  sum1, sum2, sum3 = 0., 0., 0.
  K = beta.shape[0]
  for k in range(K):
    F = beta0*E_ln_mu_k(k, beta, m, W, nu, m0)
    Eln_lam = E_ln_lam_k(k, nu, W)
    sum1 = sum1 + D*np.log(beta0/(2*np.pi)) + Eln_lam - F

    sum2 = sum2 + 0.5*(nu0-D-1)*E_ln_lam_k(k,nu,W)
    sum3 = sum3 + nu[k]*np.trace(np.dot(inv(W0),W[k]))
  lnB = ln_B(W0,nu0)
  return 0.5*sum1 + sum2 - 0.5*sum3 + K*lnB

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