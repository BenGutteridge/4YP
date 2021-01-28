# -*- coding: utf-8 -*-
"""Variational GMM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JrLIbMR4OrrBDVjIrJX8tKpMRhM-_8Wd

This notebook is based on Chapter 9 (?) of Bishop, and aims to replicate the 'variational mixture of Gaussians' example worked through there. This is to help my understanding of this example, as much of it is the basis for the gradient ascent approach used in my 4YP, and to hopefully get functioning posterior inference. I'm stuck in a rut, and better to go from a working model than try and debug a crummy bloated one built from scratch that has never worked!

"""

import autograd
import autograd.numpy as np
from autograd.numpy.linalg import inv, det, multi_dot
from autograd.scipy.special import digamma, gammaln
import os, sys
import matplotlib.pyplot as plt
from utils import *
plt.close('all')
plt.ioff()

def alpha_k(Nk, alpha0):
  return alpha0 + Nk

def beta_k(Nk, beta0):
  return beta0 + Nk

def m_k(Nk, xkbar, betak, m0, beta0):
  return ((1/betak)*(beta0*m0 + Nk*xkbar)).reshape(2)

def W_k(Nk, xkbar, Sk, m0, beta0):
  inv_Wk = inv(W0) + Nk*Sk + ((beta0*Nk)/(beta0+Nk))*np.dot((xkbar-m0),(xkbar-m0).T) 
  return inv(inv_Wk)

def nu_k(Nk, nu0):
  return nu0 + Nk

"""These use some new parameters, $N_k, \bar{x}_k, S_k$, which essentially correspond to the sum of responsibilities, responsibility-weighted mean and responsibility-weighted variance for mixture component $k$:"""

def N_k(responsibilities_k):
  return np.sum(np.array(responsibilities_k))

def x_k_bar(Nk, responsibilities_k, X):
  sum = np.zeros((1,D))
  for n in range(N):
    sum = sum + responsibilities_k[n]*X[n]
  if Nk > 0:
    return (1/Nk)*sum
  else:
    return 0.

def S_k(Nk, responsibilities_k, X, xkbar):
  sum = 0.0
  for n in range(N):
    sum = sum + responsibilities_k[n]*np.dot((X[n]-xkbar).T,(X[n]-xkbar))
  if Nk > 0:
    return (1/Nk)*sum
  else: 
    return np.eye(2) # doesn't actually matter what it returns, the component is dead

"""
These updates correspond to the M (maximisation) step of the EM algorithm. To perform them we of course need our responsibilities:
"""

def E_ln_pi_k(k, alpha):
  return digamma(alpha[k]) - digamma(np.sum(alpha))

def E_ln_lam_k(k, nu, W):
  return np.sum(digamma(nu[k]+1-np.arange(D)+1)) + D*np.log(2) + np.log(det(W[k]))

def E_ln_mu_k(k, beta, m, W, nu, xn):
  # print('m, W\n', m, W)
  return D*beta[k]**-1 + nu[k]*multi_dot(((xn-m[k]), W[k], (xn-m[k]).T))

def ln_rho_nk(k, alpha, nu, W, beta, m, xn):
  return E_ln_pi_k(k,alpha) + 0.5*E_ln_lam_k(k, nu, W) - 0.5*D*np.log(2*np.pi) - 0.5*E_ln_mu_k(k, beta, m, W, nu, xn)

"""Now we need $r_{nk}$, which we remember is $\rho_{nk}$ divided by the sum of $\rho_{nk}$ over $k$:"""

def r_nk(k, alpha, nu, W, beta, m, xn):
  rhonk = np.exp(ln_rho_nk(k, alpha, nu, W, beta, m, xn))
  sum_k_rho = 0.
  for j in range(K):
      ln_rhonk = ln_rho_nk(j, alpha, nu, W, beta, m, xn)
      if np.isnan(ln_rhonk): 
          print('\nln_rho_nk() returning nan')
          sys.exit()
      sum_k_rho = sum_k_rho + np.exp(ln_rhonk)
  if sum_k_rho>0: 
     return rhonk/sum_k_rho
  else: return 0.

"""### The algorithm
E step: Calculate responsibilities $r_{nk}$ for each and every mixture component $k$ and datapoint $x_n$.

M step: use $r_{nk}$ in the update equations for $\alpha, \beta, m, W, \nu$ to minimise the $\mathbb{KL}$ divergence between the variational distribution $q(Z,\pi,\mu,\Lambda)$ and the true posterior $p(Z,\pi,\mu,\Lambda|X)$
"""

# E step: calculate responsibility
def E_step(N,K,alpha,nu,W,beta,m,X):
  r = []
  for k in range(K):
    r.append([])
    for n in range(N):
      r[k].append(r_nk(k, alpha, nu, W, beta, m, X[n]))
  return r

# M step: update hyperparameters
def M_step(r,X,alpha0,beta0,m0,W0,nu0):
  NK, xbar, S = [],[],[]
  alpha, beta, nu = np.empty(K), np.empty(K), np.empty(K)
  m, W = [np.zeros(2) for _ in range(K)], [np.zeros((2,2)) for _ in range(K)]

  for k in range(K):
    Nk = N_k(r[k])
    xkbar = x_k_bar(Nk, r[k], X)
    Sk = S_k(Nk, r[k], X, xkbar)

    alpha[k] = alpha_k(Nk, alpha0)
    beta[k] = beta_k(Nk, beta0)
    m[k] = m_k(Nk, xkbar, beta[k], m0, beta0)
    W[k] = W_k(Nk, xkbar, Sk, m0, beta0)
    nu[k] = nu_k(Nk, nu0)

    NK.append(Nk)
    xbar.append(xkbar)
    S.append(Sk)

    # print('k=%d\nalpha'%k, alpha[k], '\nbeta', beta[k], '\nm', m[k], '\nW', W[k], '\nnu', nu[k])
  return alpha, beta, m, W, nu, NK, xbar, S

"""## Calculating ELBO
Now lets get the programme to calculate the evidence lower bound on each iteration. From Bishop, eq.10.70:
"""

# components of ELBO

def E_ln_p_X_given_Z_mu_lam(beta, m, W, nu, NK, S, xbar):
  sum = 0.
  for k in range(K):
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
  for n in range(N):
    for k in range(K):
      sum = sum + r[k][n]*E_ln_pi_k(k,alpha)
  return sum

def ln_C(alpha):
  return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

def ln_B(W,nu):
  ln_num = -0.5*nu*np.log(det(W))
  ln_det_1 = 0.5*nu*D*np.log(2) - D*(D-1)*0.25*np.log(np.pi)
  ln_det_2 = np.sum(gammaln(np.array([0.5*(nu+1-i) for i in range(D)])))
  return  ln_num - ln_det_1 - ln_det_2

def E_ln_p_pi(alpha0, alpha):
  return ln_C(alpha0*np.ones(K)) + (alpha0-1)*np.sum(np.array([E_ln_pi_k(k,alpha) for k in range(K)]))

def E_ln_p_mu_lam(beta, m, W, nu, m0, W0, nu0):
  sum1, sum2, sum3 = 0., 0., 0.
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
  for n in range(N):
    for k in range(K):
      if r[k][n] > 0: # n.b. returns nan for 0log0, so need to bypass
        sum = sum + r[k][n]*np.log(r[k][n])
  return sum

def E_ln_q_pi(alpha):
  return np.sum(np.array([(alpha[k]-1)*E_ln_pi_k(k,alpha) for k in range(K)])) + ln_C(alpha)

def H_q_lam_k(k,nu,W):
  return -ln_B(W[k],nu[k]) - 0.5*(nu[k]-D-1)*E_ln_lam_k(k, nu, W) + 0.5*nu[k]*D

def E_ln_q_mu_lam(beta,W,nu):
  sum = 0.
  for k in range(K):
    sum=sum+0.5*E_ln_lam_k(k,nu,W)+0.5*D*np.log(beta[k]/(2*np.pi))-0.5*D-H_q_lam_k(k,nu,W)
  return sum

def calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,m0,W0,nu0):
  p1 = E_ln_p_X_given_Z_mu_lam(beta, m, W, nu, NK, S, xbar)
  p2 = E_ln_p_Z_given_pi(r, alpha)
  p3 = E_ln_p_pi(alpha0, alpha)
  p4 = E_ln_p_mu_lam(beta, m, W, nu, m0, W0, nu0) # this one is an array
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

"""# Code


1.   Choose some priors, $\mathbf{\theta}_0$, set $\mathbf{\theta} = \mathbf{\theta}_0$ 
2.   Iterate until converged:   
  1.   Calculate the full set of $NK$ responsibilities
  2.   Using update equations, update each $\theta_k$
"""

from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
if not os.path.exists('plots'):
  os.mkdir('plots')

# Set priors, dataset, problem
D = 2       # dimensions
K = 3       # number of components, unnecessary ones should go to zero
N = 100     # number of points in synthetic dataset
N_its = 0 # number of updates

# Dataset
centres = [np.array([0.,8.]), np.array([5.,0.])]
covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])] 
X1 = multivariate_normal(mean=centres[0],
                         cov=covs[0],
                         size=int(N/2))
X2 = multivariate_normal(mean=centres[1],
                         cov=covs[1],
                         size=int(N/2))
X = np.concatenate((X1,X2))

# Variational priors
alpha0 = 1e-3     # as alpha0 -> 0, pi_k -> 0. As alpha0 -> Inf, pi_k -> 1/K
beta0 = 1e-10     # ???
m0 = np.zeros(2)  # zero by convention (symmetry)
W0 = np.eye(2)    # 
nu0 = 2           #


# r needs to be randomised (not uniform) because if its all the same nothing 
# changes - not sure of the mathematical reasoning for this
r = np.array([np.random.dirichlet(np.ones(K)) for _ in range(N)])
r = [r[:,k] for k in range(K)]

# neaten up the output
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

verbose = False

ELBO, ELBO_M, ELBO_E = np.empty(2*N_its), np.empty(N_its), np.empty(N_its)
for i in tqdm(range(N_its)):
  # M step
  alpha, beta, m, W, nu, NK, xbar, S = M_step(r,X,alpha0,beta0,m0,W0,nu0)

  def E_pi(alpha, alpha0, N):
    return [(alpha[k])/(K*alpha0 + N) for k in range(K)]
  
  Epi = E_pi(alpha, alpha0, N)
  ELBO[2*i] = calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,m0,W0,nu0)
  ELBO_M[i] = ELBO[2*i]
  
  if verbose:
    print('\n******************Iteration %d************************\n'%i)
    print('\nalpha', alpha, '\nbeta', beta, '\nnu', nu, '\nm', m, '\nW', W, '\nnu', nu)
    print('E[pi] = ', Epi)
    print('ELBO = %f'%ELBO[i])

  # Plot
  title = 'iteration %d' % i
  filename = 'plots/img%04d.png'%i
  # plot_GMM(X, mu, lam, pi, centres, covs, K, title)
  plot_GMM(X, m, inv(S), Epi, centres, covs, K, title, savefigpath=filename)

  # E step
  r = E_step(N,K,alpha,nu,W,beta,m,X)
  ELBO[2*i+1] = calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,m0,W0,nu0)
  ELBO_E[i] = ELBO[2*i+1]

# Make and display gif 
filedir = 'plots'
gifdir = 'gifs'
gifname = make_gif(filedir, gifdir)

# delete pngs for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))

Image(open("gifs/%s.gif"%gifname,'rb').read())

fig=plt.figure(figsize=(14,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(np.arange(0,N_its,0.5), ELBO)
plt.plot(ELBO_M)
plt.plot(np.arange(N_its)+0.5, ELBO_E)
plt.legend(['overall', 'after M step', 'after E step'])
plt.xlabel('Iterations')
plt.ylabel('Evidence lower bound')
plt.show();

"""## Gradient ascent optimisation of ELBO

From Bishop:

*Finally, it is worth noting that the lower bound provides an alternative approach for deriving the variational re-estimation equations ... To do
this we use the fact that, since the model has conjugate priors, the functional form of the factors in the variational posterior distribution is known, namely discrete for $Z$, Dirichlet for $\pi$, and Gaussian-Wishart for $(\mu_k,\lambda_k)$. By taking general parametric forms for these distributions we can derive the form of the lower bound as a function of the parameters of the distributions. Maximizing the bound with respect to these parameters then gives the required re-estimation equations.*

i.e. ELBO is a function of the distributional parameters $\mathbf{\theta} = \alpha,\beta,m,W,\nu$. Therefore we can calculate $\nabla_\theta(ELBO)$ and perform gradient ascent to maximise it and find the variational parameters of $q$, rather than using iterative update equations.
"""

def ELBO_theta(a,b,m,V,u,X,alpha0,m0,W0,nu0):
  alpha = a**2
  beta = b**2
  W = [np.dot(V[k],V[k].T) for k in range(K)]
  nu = abs(u) + 2
    
  r, NK  = [],[]
  xbar, S = [],[] 
  for k in range(K):
    r.append([])
    for n in range(N):
      r[k].append(r_nk(k, alpha, nu, W, beta, m, X[n]))
      if np.isnan(r[k][-1]):
          print('Everything: ', k,alpha,nu,W,beta,m,X[n])
          return 0
  for k in range(K):
    NK.append(N_k(r[k]))
    xbar.append(x_k_bar(NK[k], r[k], X))
    S.append(S_k(NK[k], r[k], X, xbar[k]))
  return calculate_ELBO(r, alpha, beta, m, W, nu, NK, S, xbar, alpha0, m0, W0, nu0)

def get_S(a,b,m,V,u,X):
  alpha = a**2
  beta = b**2
  W = [np.dot(V[k],V[k].T) for k in range(K)]
  nu = abs(u) + 2
    
  r, NK  = [],[]
  xbar, S = [],[] 
  for k in range(K):
    r.append([])
    for n in range(N):
      r[k].append(r_nk(k, alpha, nu, W, beta, m, X[n]))
  for k in range(K):
    NK.append(N_k(r[k]))
    xbar.append(x_k_bar(NK[k], r[k], X))
    S.append(S_k(NK[k], r[k], X, xbar[k]))
  return S

# ELBO_theta(alpha,beta,m,W,nu,X,alpha0,m0,W0,nu0)

"""The problem at the moment is that autograd does not support indexing, e.g. `r[n,k] = XXX`. I have instead swapped to `r[k][n]`, i.e. a list of K lists of length N, building it by `.append` rather than assignment."""

# setup stuff
from autograd import grad
r = np.array([np.random.dirichlet(np.ones(K)) for _ in range(N)])
r = [r[:,k] for k in range(K)]
alpha, beta, m, W, nu, NK, xbar, S = M_step(r,X,alpha0,beta0,m0,W0,nu0)
a, b, u = alpha**0.5, beta**0.5, nu-2
V = np.linalg.cholesky(W)

N_its = 10
step_a, step_b, step_m, step_V, step_u = 1e-1,1e-1,1e-2,1e-3,1e-2
ELBO = np.zeros(N_its+1)
ELBO[0] = ELBO_theta(a, b, m, V, u, X, alpha0, m0, W0, nu0)


# Iterate
for i in range(N_its):
    print('\n*****************')
    
    
    D_a = grad(ELBO_theta, 0)
    d_a = D_a(a, b, m, V, u, X, alpha0, m0, W0, nu0)
    try: d_a = d_a._value
    except: 
        print('ERROR: a is type ', type(d_a))
        break
    a = a + step_a*d_a
    print('\nalpha: ', a**2)
      
    D_b = grad(ELBO_theta, 1)
    d_b = D_b(a, b, m, V, u, X, alpha0, m0, W0, nu0)
    try: d_b = d_b._value
    except: 
        print('ERROR: b is type ', type(d_b))
        break
    b = b + step_b*d_b
    print('\nbeta: ', b**2)
      
    D_m = grad(ELBO_theta, 2)
    d_m = D_m(a, b, m, V, u, X, alpha0, m0, W0, nu0)  
    try: d_m = d_m._value
    except: 
        print('ERROR: m is type ', type(d_m))
        break
    m = [m[k] + step_m*d_m[k] for k in range(K)]
    print('\nm: ', m)
    
    D_V = grad(ELBO_theta, 3)
    d_V = D_V(a, b, m, V, u, X, alpha0, m0, W0, nu0)
    try: d_V = d_V._value
    except: 
        print('ERROR: d_V is type ', d_V) # we know it's a list - doesn't seem to have any arrayboxes
        break
    print('d_V: ', d_V)
    if np.isnan(d_V[0][0,0]):
        print('\nBROKEN\nELBO: ', ELBO_theta(a, b, m, V, u, X, alpha0, m0, W0, nu0))
        print('Grad: ', D_V(a, b, m, V, u, X, alpha0, m0, W0, nu0)._value)
        break
    V = [V[k] + step_V*d_V[k] for k in range(K)]
    # print('new V: ', V )
    print('\nW: ', [np.dot(V[k],V[k].T) for k in range(K)])
    
    D_u = grad(ELBO_theta, 4)
    d_u = D_u(a, b, m, V, u, X, alpha0, m0, W0, nu0)
    try: d_u = d_u._value
    except: 
        print('ERROR: d_u is type ', type(d_u) )# we know it's a list - doesn't seem to have any arrayboxes
        break
    u = u + step_u*d_u
    print('\nnu: ', abs(u)+2)
    
    # Plot
    title = 'iteration %d' % i
    filename = 'plots2/img%04d.png'%i
    # plot_GMM(X, mu, lam, pi, centres, covs, K, title)
    
    S = get_S(a,b,m,V,u,X)
    Epi = np.array([alpha[k]/np.sum(alpha) for k in range(K)]) # expectation of Wishart is alpha_k/sum(alpha)
    plot_GMM(X, m, inv(S), Epi, centres, covs, K, title, savefigpath=filename)
     
    ELBO[i+1] = ELBO_theta(a, b, m, V, u, X, alpha0, m0, W0, nu0)
    print('\n**%d**\nOld ELBO: %.4f\nNew ELBO: %.4f\nIncrease = %.4f, '%(i, ELBO[i], ELBO[i+1], ELBO[i+1]-ELBO[i]), (ELBO[i+1]-ELBO[i])>0)

# Make and display gif 
filedir = 'plots2'
gifdir = 'gifs2'
gifname = make_gif(filedir, gifdir)

# delete pngs for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))

Image(open("gifs2/%s.gif"%gifname,'rb').read())

