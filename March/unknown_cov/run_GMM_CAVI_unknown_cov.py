# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:54:39 2021

@author: benpg

CAVI: unchanged, unknown covariance
"""

# General libraries
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

# Own functions
from generate_dataset import generate_2D_dataset
from calculate_responsibilities_unknown_cov import * 
from plot_utils import *
from EM_steps_unknown_cov import E_step, M_step
from calculate_ELBO_unknown_cov import calculate_ELBO

# Dataset and update params
N_its = 100
N = 500
K = 20
D = 2

N_clusters = 5
X, centres, covs = generate_2D_dataset(N, K=N_clusters)

# Variational priors
alpha0 = 1e-3     # as alpha0 -> 0, pi_k -> 0. As alpha0 -> Inf, pi_k -> 1/K
beta0 = 1e-10     # ???
m0 = np.zeros(2)  # zero by convention (symmetry)
W0 = np.eye(2)    # 
nu0 = 2           # 

# Saving plots and making gif
filedir = 'plots'
gifdir = 'gifs'


r = np.array([np.random.dirichlet(np.ones(K)) for _ in range(N)])
r = [r[:,k] for k in range(K)]

# for plotting
alphas, NKs, betas = np.zeros((N_its,K)), np.zeros((N_its,K)), np.zeros((N_its,K))
ms = np.zeros((N_its,K,D))
varx, vary, covxy = np.empty((N_its,K)),np.empty((N_its,K)),np.empty((N_its,K)) 

verbose = False

ELBO, ELBO_M, ELBO_E = np.empty(2*N_its), np.empty(N_its), np.empty(N_its)
for i in tqdm(range(N_its)):
    
  # M step
  alpha, beta, m, W, nu, NK, xbar, S = M_step(r,X,alpha0,beta0,m0,W0,nu0,K)
  ELBO[2*i] = calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,beta0,m0,W0,nu0)
  ELBO_M[i] = ELBO[2*i]
 
  # E step
  r = E_step(N,K,alpha,nu,W,beta,m,X)
  ELBO[2*i+1] = calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,beta0,m0,W0,nu0)
  ELBO_E[i] = ELBO[2*i+1]
  
  # Plot stuff
  alphas[i,:] = alpha
  NKs[i,:] = NK
  betas[i,:] = beta
  # m is shape K,D when cast to nparray
  ms[i,:,:] = np.array(m) # shape N_its*K*D
  varx[i,:] = np.array([inv(S[k])[0,0] for k in range(K)]) 
  vary[i,:] = np.array([inv(S[k])[1,1] for k in range(K)]) 
  covxy[i,:] = np.array([inv(S[k])[1,0] for k in range(K)]) 

  # Plot
  Epi = E_pi(alpha, alpha0, N)
  title = 'CAVI: iteration %d' % i
  filename = 'plots/img%04d.png'%i
  # plot_GMM(X, mu, lam, pi, centres, covs, K, title)
  plot_GMM(X, m, inv(S), Epi, centres, covs, K, title, savefigpath=filename)
  
  if verbose:
      print('\n******************Iteration %d************************\n'%i)
      print('\nalpha', alpha, '\nbeta', beta, '\nnu', nu, '\nm', m, '\nW', W, '\nnu', nu)
      print('E[pi] = ', Epi)
      print('ELBO = %f'%ELBO[i])
  
# Make and display gif 
gifname = make_gif(filedir, gifdir)
# delete pngs for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))
  
  
# Plot parameters
plt.close('all')
plot_ELBO(ELBO, ELBO_E, ELBO_M, N_its)
plot_1D_phi(alphas, 'alphas', K)
plot_1D_phi(betas, 'betas', K)
plot_1D_phi(NKs, 'Nk', K)
plot_K_covs(varx, vary, covxy, K)
plt.show()
    
    




