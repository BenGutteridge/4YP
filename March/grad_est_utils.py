# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:34:29 2021

@author: benpg
"""
import numpy as np
from numpy import dot
from numpy.linalg import inv, det, multi_dot
from scipy.stats import gamma
from scipy.special import digamma
from scipy.special import polygamma as psi_prime
from calculate_ELBO import ln_C
from scipy.stats import gamma
from copy import copy
    
def sample_mu(var, D=2):
    mu = np.zeros((var.K,D))
    for k in range(var.K):
        mu[k] = np.random.multivariate_normal(var.means[k], var.covariances[k])
    return mu
        
def sample_pi(var):
    return np.random.dirichlet(var.alpha)
    
def grad_ln_q_pi(var, pi):
    alpha = var.alpha
    grad = np.zeros(alpha.shape[0])
    for k in range(alpha.shape[0]):
        grad[k] = digamma(np.sum(alpha)) - digamma(alpha[k]) + alpha[k]*np.log(pi[k])
    return grad

def grad_ln_q_mu(var, mu):
    m, C, invC = var.means, var.covariances, var.precisions
    K, D = m.shape[0], m.shape[1]
    grad_m, grad_C = np.zeros((K,D)), np.zeros((K,D,D))
    for k in range(K):
        v = (mu[k] - m[k]).reshape(2,1)
        grad_m[k] = -1 * np.dot(invC[k], v).reshape(D,)
        grad_C[k] = -0.5 * (np.dot(v, v.T) + invC[k])
    return grad_m, grad_C

    
def grad_H_q(var):
    K, D = var.K, var.means.shape[1]
    alpha = var.alpha
    invC = var.precisions
    
    # GRAD E_q[ln q(Z) + ln q(pi) + ln q(mu)]
    alpha_hat = np.sum(alpha)
    nabla_alpha_H = np.zeros(K)
    nabla_m_H = np.zeros((K,D)) # won't change
    nabla_C_H = np.zeros((K,D,D))
    for k in range(K):
        b = psi_prime(1, alpha[k]) - psi_prime(1, alpha_hat)
        c = digamma(alpha[k]) - digamma(alpha_hat)
        nabla_alpha_H[k] = (alpha[k] - 1) * b + c
        # entropy of a multivariate gaussian is D/2 ln (2 pi e) + 0.5 ln|C_k|
        nabla_C_H[k] = 0.5 * invC[k]
    nabla_m_H = nabla_m_H
    return nabla_alpha_H, nabla_m_H, nabla_C_H


def ln_p_X_given_Z_mu(var, X, mu):
    D, N, K = X.shape[1], X.shape[0], mu.shape[0]
    c = D*np.log(2*np.pi) + np.log(det(var.inv_sigma))
    r = var.responsibilities
    _sum = 0.
    for n in range(N):
        for k in range(K):
            v = (X[n] - mu[k]).reshape(2,1)
            _sum += r[n,k] * (multi_dot((v.T, var.inv_sigma, v)) + c)
    return -0.5 * _sum # should be -1*

def ln_p_Z_given_pi(var, pi):
    N, K = var.means.shape[1], var.K
    _sum = 0.
    for n in range(N):
        for k in range(K):
            _sum += var.responsibilities[n,k] * np.log(pi[k])
    return _sum

def ln_p_pi(var, joint, pi):
    lnC = ln_C(np.ones(var.K)*joint.alpha)
    return lnC + (joint.alpha - 1) * np.sum(np.log(pi))

def ln_p_mu(var, joint, mu):
    D = mu.shape[1]
    c = D*np.log(2*np.pi) + np.log(det(joint.covariance))
    _sum = 0.
    for k in range(var.K):
        v = (mu[k] - joint.mean).reshape(2,1)
        _sum += multi_dot((v.T, joint.precision, v)) + c
    return -0.5 * _sum

def evaluate_cost_SFE(var, joint, X, pi_hat, mu_hat):
    # f = ln p(X|Z,mu) + ln p(Z|pi) + ln p(pi) + ln p(mu)
    # this is a black box
    a = ln_p_X_given_Z_mu(var, X, mu_hat)
    b = ln_p_Z_given_pi(var, pi_hat)
    c = ln_p_pi(var, joint, pi_hat)
    d = ln_p_mu(var, joint, mu_hat)
    # print(a,b,c,d)
    return a + b + c + d



# %% Pathwise estimator functions

### Pathwise samples
def sample_pathwise_pi(var):
    alpha = var.alpha
    u = np.random.rand(alpha.shape[0])
    z = gamma.ppf(u, alpha, scale=1) # inv CDF Gamma
    pi = z/np.sum(z)
    return pi, z, u

def sample_pathwise_mu(var):
    m, C = var.means, var.covariances
    K, D = var.K, m.shape[1]
    xi = np.random.normal(size=(K,D))
    mu = np.zeros((var.K,D))
    for k in range(K):
        mu[k] = m[k] + xi[k] * np.diag(C[k])**0.5
    return mu, xi

### pi (dirichlet)
def grad_alpha_f(var, pi, z, u):
    # here z denotes a Beta sample (will be transformed to Dirichlet)
    r = var.responsibilities
    D_pi_f = var.alpha + np.sum(r, axis=0) / pi
    D_z_pi = grad_z_pi(var, pi, z)
    eps = 1e-5
    D_alpha_invF = np.zeros(var.K) # this is the inverse CDF Gamma dif wrt alpha
    for k in range(var.K):
        D_alpha_invF[k] = (1/eps) * (gamma.ppf(u[k], var.alpha[k] + eps) - gamma.ppf(u[k], var.alpha[k]))
    return D_pi_f * D_z_pi * D_alpha_invF

def grad_z_pi(var, pi, z):
    K = var.K
    D_z_pi = np.zeros(K)
    for k in range(K):
        z_not_k = copy(z)
        z_not_k[k] = 0.
        D_z_pi[k] = np.sum(z_not_k) / np.sum(z)**2
    return D_z_pi


### mu (normal), m
def grad_m_f(var, joint, X, sample_mu):
    D_mu_f = grad_mu_f(var, joint, X, sample_mu)
    D_m_mu = 1.
    return D_mu_f * D_m_mu

def grad_mu_f(var, joint, X, mu):
    K,D = var.K, X.shape[1]
    D_mu_f = np.zeros((K,D))
    for k in range(K):
        a = np.zeros(D,)
        for n in range(X.shape[0]):
            a += var.responsibilities[n][k] * dot((X[n] - mu[k]), var.inv_sigma)
        b = dot((mu[k] - joint.mean), joint.precision)
        c = dot((mu[k] - var.means[k]), var.precisions[k])
        D_mu_f[k] = a + b - c
    return D_mu_f

### mu (normal), C
def grad_rootC_f(var, joint, X, sample_xi, sample_mu):
    D_mu_f = grad_mu_f(var, joint, X, sample_mu)
    D_rootC_mu = sample_xi
    grad_C_f =  D_mu_f * D_rootC_mu
    return np.array([np.diag(grad_C_f[k]) for k in range(var.K)])


from scipy.special import logsumexp
### Calculate IWAE weights
def calculate_iwae_weights(n_samples, var, joint, X, pi_samples, mu_samples):
    # evaluate IWAE weights for gradient estimator
    ln_p_over_q_samples = np.zeros(n_samples)
    for i in range(n_samples):
        ln_p_over_q_samples[i] = evaluate_ln_p_over_q(var, joint, X, 
                                                pi_samples[i], mu_samples[i])
    ln_weights = ln_p_over_q_samples - logsumexp(ln_p_over_q_samples)
    print(ln_weights)
    return np.exp(ln_weights)

#evlauate p/q for the purposes of IWAE weights
def evaluate_ln_p_over_q(var, joint, X, pi, mu):
    # ln(p/q) = lnp - lnq
    lnp = evaluate_cost_SFE(var, joint, X, pi, mu)
    lnq = ln_q(var, pi, mu)
    return lnp - lnq
    
from calculate_ELBO import E_ln_q_Z
def ln_q(var, pi, mu):
    D = mu.shape[1]
    a = E_ln_q_Z(var.responsibilities)
    b = ln_C(var.alpha) + np.sum((var.alpha - 1)*np.log(pi))
    c = 0.
    for k in range(var.K):
        v = (mu[k] - var.means[k]).reshape(2,1)
        c += -0.5 * (multi_dot((v.T, var.precisions[k], v)) + D*np.log(2*np.pi)
                     + np.log(det(var.covariances[k])))
    return a + b + float(c)

    
    
