# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:04:19 2021

@author: benpg

testing gradient
"""
from autograd import grad
import autograd.numpy as np
from utils import calculate_ELBO

# step = 0.001

def testing_autograd(x,Z,pi,mu,lam,new_a,b,m,V,u,p_dist_params,K,N):
    for i in range(2):
        for j in range(2):
            L = calculate_ELBO(x,Z,pi,mu,lam,new_a,b,m,V,u,p_dist_params,K,N)
            
            mplusdm, mminusdm = np.copy(m), np.copy(m)
            mplusdm[i][j] = m[i][j] + 0.001
            mminusdm[i][j] = m[i][j] - 0.001
    
            
            LplusdL = calculate_ELBO(x,Z,pi,mu,lam,new_a,b,mplusdm,V,u,p_dist_params,K,N)
            LminusdL = calculate_ELBO(x,Z,pi,mu,lam,new_a,b,mminusdm,V,u,p_dist_params,K,N)
            
            num_gradL = (LplusdL - LminusdL)/(2*(1/1000))
            
            Lgrad_m = grad(calculate_ELBO, 7)
            gradL = Lgrad_m(x,Z,pi,mu,lam,new_a,b,m,V,u,p_dist_params,K,N)
            
            print('Numerical gradient %d= '%(i+2*j), num_gradL)
            
    print('Gradient with autograd = ', gradL)
        
        # # updating
        # new_m = []
        # for ms in m:
        #     new_m.append(m + step*gread=)   