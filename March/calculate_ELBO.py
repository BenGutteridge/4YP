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
from copy import copy

def E_ln_p_X_given_Z_mu(m, invSig, C0, NK, xbar, SK, D=2):
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
  if isinstance(r, list): r = np.array(r).T
  N = r.shape[0]
  for n in range(N):
    for k in range(alpha.shape[0]):
      sum = sum + r[n,k]*E_ln_pi_k(k,alpha)
  return sum

def ln_C(alpha):
    # for dirichlet, not to be confused with C for covariance
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
    Ksum = 0.0
    for k in range(len(m)):
        a = -D*np.log(np.pi*2)
        b = np.log(det(invC0))
        v = (m[k]-m0).reshape(2,1)
        c = -multi_dot((v.T, invC0, v)) - np.trace(np.dot(invC0, C[k]))
        Ksum +=  0.5*(a+b+c)
    return float(Ksum)

def E_ln_q_Z(r):
    # convert r into an NxK array if it is a list
    if isinstance(r, list): r = np.array(r).T
    N, K = r.shape[0], r.shape[1]
    sum = 0.
    for n in range(N):
        for k in range(K):
            if r[n,k] > 0: # n.b. returns nan for 0log0, so need to bypass
                sum = sum + r[n,k]*np.log(r[n,k])
    return sum

def E_ln_q_pi(alpha):
    K = alpha.shape[0]
    return np.sum(np.array([(alpha[k]-1)*E_ln_pi_k(k,alpha) for k in range(K)])) + ln_C(alpha)

# def H_q_lam_k(k,nu,W,D=2):
#   return -ln_B(W[k],nu[k]) - 0.5*(nu[k]-D-1)*E_ln_lam_k(k, nu, W) + 0.5*nu[k]*D

# def E_ln_q_mu_lam(beta,W,nu,D=2):
#   sum = 0.
#   for k in range(beta.shape[0]):
#     sum=sum+0.5*E_ln_lam_k(k,nu,W)+0.5*D*np.log(beta[k]/(2*np.pi))-0.5*D-H_q_lam_k(k,nu,W)
#   return sum

def E_ln_q_mu(C, D=2): # n.b. this is entropy of q(mu_k)
    Ksum = 0.0
    for k in range(len(C)):
        Ksum += 0.5*D*(np.log(2*np.pi) + 1) + 0.5*np.log(det(C[k]))
    return Ksum


def calculate_ELBO(r, alpha, m, C, invSig, alpha0, m0, C0, NK, xbar, SK):
      p1 = E_ln_p_X_given_Z_mu(m, invSig, C0, NK, xbar, SK, D=2)
      p2 = E_ln_p_Z_given_pi(r, alpha)
      p3 = E_ln_p_pi(alpha0, alpha)
      p4 = E_ln_p_mu(m, C, m0, inv(C0), D=2)
      q1 = E_ln_q_Z(r)
      q2 = E_ln_q_pi(alpha)
      q3 = E_ln_q_mu(C, D=2)    
      return p1+p2+p3+p4-q1-q2-q3
  
def calculate_ELBO_variance(var, alpha0, m0, C0, samples, X):
    # calculate the variance of the ELBO estimate calculated using minibatches
    alpha, m, C, invSig = var.alpha, var.means, var.covariances, var.inv_sigma
    r = var.responsibilities
    N, K = X.shape[0], alpha.shape[0]
    empty_r, j = np.zeros((N,K)), 0
    ELBOs = np.zeros(len(samples))
    for i in samples:
        r_i = copy(empty_r)
        r_i[:,:] = r[i]
        ELBOs[j] = calculate_ELBO(r_i, alpha, m, C, invSig, alpha0, m0, 
                                  C0, N*r[i], xbar=X[i]*np.ones((K,2)), SK=np.zeros((K,2,2)))
        E_ELBO = np.mean(ELBOs)
        Var_ELBO = np.mean((ELBOs - E_ELBO)**2)
        j += 1
    return E_ELBO, Var_ELBO
    