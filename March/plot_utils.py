# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:21:20 2021

@author: benpg

Plot utility functions
"""

import matplotlib as mpl
import sys
from IPython.display import Image
import imageio
import os
import datetime
import matplotlib.pyplot as plt
from scipy.stats import chi2
# import autograd.numpy as np
# from autograd.numpy.linalg import det, inv
import numpy as np
from numpy.linalg import det, inv
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

all_cols_m = ['m' for _ in range(200)]

def make_gif(filedir, gifdir, gifname=''):
  gifname = str(datetime.datetime.now())[:-7].replace(':',';') + '__' + gifname
  with imageio.get_writer(gifdir+'/'+gifname+'.gif', mode='I') as writer:
      for filename in sorted(os.listdir(filedir)):
          image = imageio.imread(os.path.join(filedir,filename))
          writer.append_data(image)

  return gifname

def draw_ellipse(mu, cov, conf=.95):
    chiscale = chi2.isf(1-conf,2)
    v, w = np.linalg.eigh(cov)
    v = 2. * np.sqrt(chiscale) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mu, v[0], v[1], 180. + angle)
    return ell

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
# Expected value of mixture coefficients pi_k 
# for plotting ellipses, relative weights, used for transparency
def E_pi(alpha):
    return alpha/np.sum(alpha)

def plot_GMM(X, mu, lam, pi, centres, covs, K, title, savefigpath=False, xylims=[-5,10,-5,15], cols=cols):
    plt.figure(figsize=(4,4))
    if xylims != None:
        plt.xlim(xylims[0],xylims[1])
        plt.ylim(xylims[2],xylims[3])
    plt.plot(X[:,0], X[:,1], 'kx', alpha=0.2)
    
    legend = ['Datapoints']
    
    # label clusters if too many to have unique colours
    if K > len(cols):
        for k in range(K):
            plt.text(mu[k][0],mu[k][1], 'k=%d'%k)
        cols = 'r'*200
        
    for k in range(K):
        if pi[k] > 1e-3:
            plt.plot(mu[k][0], mu[k][1], cols[k], marker='o', linestyle=None)
        else: 
            plt.plot(mu[k][0], mu[k][1], cols[k], marker='X', linestyle=None)

        cov = inv(lam[k])
        ell = draw_ellipse(mu[k], cov)
        ell.set_alpha(pi[k])
        # ell.set_edgecolor('m')
        ell.set_fill(True)
        splot = plt.subplot(1, 1, 1)
        splot.add_artist(ell)
        
    # Plotting the ellipses for the GMM that generated the data
    if centres is not None and covs is not None:
        for i in range(len(centres)):
            true_ell = draw_ellipse(centres[i], covs[i])
            true_ell.set_edgecolor('g')
            true_ell.set_fill(False)
            splot.add_artist(true_ell)
    
    plt.title(title)
    # plt.legend(legend)
    if isinstance(savefigpath, str):
        plt.savefig(savefigpath)
        # plt.savefig(savefigpath[:-4]+'.pdf', format='pdf')
        # plt.close('all')
    else:
        plt.show()
    

def plot_ELBO(ELBO, ELBO_E, ELBO_M, N_its):
    fig=plt.figure(figsize=(14,6), dpi= 100, facecolor='w', edgecolor='k')
    plt.plot(np.arange(0,N_its,0.5), ELBO)
    plt.plot(ELBO_M)
    plt.plot(np.arange(N_its)+0.5, ELBO_E)
    plt.legend(['overall', 'after M step', 'after E step'])
    plt.xlabel('Iterations')
    plt.ylabel('Evidence lower bound')
    plt.show();
    
    
# Plot evolution of a 1D parameter (a K-length vector) evolve over time 
def plot_1D_phi(phis, title, K, savefigdir=None):
    plt.figure()
    legends = []
    for k in range(K):
      plt.plot(phis[:,k].T)
      legends.append('k=%d'%k)
    plt.legend(legends)
    plt.title(title)
    plt.xlabel('Iterations')
    if savefigdir is not None:
        plt.savefig(os.path.join(savefigdir, title))
 
# Plot individual elements (variance in x,y, and covariance) of K components over time
def plot_K_covs(varx,vary,covxy,K):
    legends = []
    for k in range(K):
        plt.figure()
        plt.plot(varx[:,k].T, 'r')
        plt.plot(vary[:,k].T, 'b')
        plt.plot(covxy[:,k].T, 'g')
        legends += ['var_x', 'var_y', 'cov_xy']
        plt.legend(legends)
        plt.title('Covariance matrix k=%d'%k)
        plt.xlabel('Iterations')
        
def plot_cov_ellipses_gif(variational_memory, N_its, K, gifname, xylims=[-5,10,-5,15], savefigpath='plots'):
    for n in range(1,N_its):
        legend = []
        mean = variational_memory[n].means
        cov = variational_memory[n].covariances
        weight = variational_memory[n].mixing_coefficients
        
        plt.figure()
        plt.title('Iteration %d'%n)
        if xylims != None:
            plt.xlim(xylims[0],xylims[1])
            plt.ylim(xylims[2],xylims[3])
       
        for k in range(K):
            if weight[k] > 1e-3:
                plt.plot(mean[k][0],mean[k][1],'o')
            else: 
                plt.plot(mean[k][0],mean[k][1],'X')
            # plt.text(mean[k][0],mean[k][1], 'k=%d'%k)
            legend.append('k=%d'%k)
            ell = draw_ellipse(mean[k], cov[k])
            ell.set_alpha(weight[k])
            # ell.set_edgecolor('m')
            ell.set_fill(True)
            splot = plt.subplot(1, 1, 1)
            splot.add_artist(ell)
        
        plt.legend(legend)
        if isinstance(savefigpath, str):
            plt.savefig(os.path.join(savefigpath, '%04d.png'%n))
        else:
            plt.show()
    make_gif(filedir=savefigpath, gifdir='gifs', gifname=gifname)
    
    for file in os.listdir(savefigpath):
        os.remove(os.path.join(savefigpath,file))
    
    
           
      
        		