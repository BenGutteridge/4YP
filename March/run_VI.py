"""

Main file running VI on a GMM

Ben Gutteridge

"""
# %% Import libraries and own functions/classes

import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from tqdm import tqdm
import os
import datetime
import matplotlib.pyplot as plt
# Close plots and turn off automatic figure creation (quirk of Spyder)
plt.close('all')
plt.ioff()  

from distribution_classes import JointDistribution, VariationalDistribution
from generate_dataset import generate_2D_dataset
from plot_utils import plot_GMM, E_pi, make_gif, plot_1D_phi as plot_1D_param, plot_K_covs, plot_cov_ellipses_gif

from os.path import join
import pickle
import time
from log_likelihood import log_likelihood



for i in range(1):
    seed = np.random.randint(100)
    np.random.seed(seed)
    
    t0 = time.time()
    
    # %% SAVING STUFF SETUP
    
    # update_type = 'GD'      # Using (true) gradient descent
    update_type = 'CAVI'  # Using co-rdinate ascent variational inference algo
    # update_type = 'SGD'
    # update_type = 'SNGD'  # SVI, stochastic natural gradients
    # update_type = 'SFE'   # Using regular E-step and score function estimator for phi grads
    # update_type = 'PW'      # Using regular E-step and pathwise grad est
    
    root = r"C:\Users\benpg\Documents\4YP\Running_experiments"
    run_dir = join(root, str(datetime.datetime.now())[:-7].replace(':',';') + 
                   ' --- ' + update_type)
    os.mkdir(run_dir)
    gifplots_dir = join(run_dir, 'gif_plots')
    os.mkdir(gifplots_dir)
    
    "For plotting"
    variational_memory = []
    filedir = gifplots_dir
    gifdir = run_dir

    
    # %% Misc setup
    
    "Iterations and update_type"
    K = 9                   # Initial number of mixture components
    N_its = 5             # Number of iterations of chosen update method performed
    
    
    print('\n%s' % update_type)
    minibatch_size = 20
    
    "Define parameters of joint distribution (effectively priors on variational params)"
    alpha_0 = 1e-3      # Dirichlet prior p(pi) = Dir(pi|alpha0)
    m_0 = np.zeros(2)   # Gaussian prior p(mu) = N(mu|m0, C0)
    C_0 = np.eye(2)     # Covariance of prior mu
    inv_sigma = np.eye(2)   # Fixed covariance/precision of Gaussian components 
    K_inv_sigma = [inv_sigma for _ in range(K)] # for plotting
    
    
    "Generating dataset"
    N = 500
    num_clusters = 7
    centres = np.array([[6,3], [3,6], [12,15], [9,6], [3,14], [9,11], [13,1]])
    covs = np.array([np.eye(2) for _ in range(num_clusters)])
    weights = np.ones(num_clusters)/num_clusters
    X, centres, covs, weights = generate_2D_dataset(N, K=num_clusters,
                                           # weights=np.random.dirichlet(np.ones(num_clusters)),
                                           centres=centres, covs=covs,
                                           weights=weights)
    
    true_log_likelihood = log_likelihood(X, weights, centres, covs)
    
    # # trying with a real dataset
    # X = np.loadtxt(r"C:\Users\benpg\Documents\4YP\Datasets\s1.txt")
    # X = X/2.5e4
    # N = X.shape[0]
    # K = 20
    # K_inv_sigma = [inv_sigma for _ in range(K)]
    # centres, covs = None, None
    
    """Schedule for step sizes in GD. Constant by default (no t input) or a decaying
    step size. forgetting rate is between 0.5 and 1 and indicates how quickly old
    info is forgotten, delay >= 0 and downweights early iterations."""
    scale, delay, forgetting = 1, 1, 0.5
    step_sizes = {'alpha': 1, 'm': 1e-3, 'C': 1e-3, 'lam1': 1e-4, 'lam2': 1e-4}
    def gd_schedule(t=None, scale=scale, delay=delay, forgetting=forgetting,
                    step_sizes=step_sizes): # maybe use kwargs?
        if t is None: return step_sizes
        else:    
            # rho_t = scale*(t + delay)**(-forgetting) # Eq 26, Hoffman SVI
            # rho_t = scale*np.exp(-0.05*t)   # off the top of my head
            rho_t = 1                       # constant step size
            # decay, A = 1, 0               # A is stability constant >= 0
            # rho_t = scale/(t+1+A)**decay  # See Spall 4.14
            steps= {}
            for key in step_sizes:
                if update_type == 'SNGD':
                    steps[key] = 1*rho_t
                else:
                    steps[key] = step_sizes[key] * rho_t 
            return steps
    
    "Initialising params and instantiating distributions"
    # TODO: could we move this into VariationalDistribution? Should we avoid dependence on N?
    
    initial_responsibilities = np.random.dirichlet(np.ones(K), N)
    
    variational = VariationalDistribution(initial_responsibilities, 
                                          inv_sigma, update_type, gd_schedule)
    joint = JointDistribution(alpha_0, m_0, covariance=C_0)
    variational.initialise_params(X)
    
    
    # %% Implement update steps
    
    "Repeat Maximisation and Expectation steps for N_its iterations, maximising ELBO"
    t_start, t_end = np.zeros(N_its), np.zeros(N_its)
    for i in tqdm(range(N_its)):
        variational.calculate_weighted_statistics(X)
        variational.calculate_ELBO(joint)
        print(variational.ELBO)
        # stop if ELBO suddenly goes nuts
        if i > 15:
            if abs(variational.ELBO) > abs(variational_memory[-10].ELBO):
                print('ELBO gone catastrophically large, ELBO = ', variational.ELBO)
                N_its = i
                break
        # Plotting stuff
        title = '%s: Iteration %d -- ELBO = %7.0f'%(variational.update_type, i, variational.ELBO)
        if update_type == 'SGD' or update_type == 'SNGD':
            title = 'Minibatch: %d, ' % minibatch_size + title
        filename = join(gifplots_dir, 'img%04d.png'%i)
        variational.calculate_mixing_coefficients()
        plot_GMM(X, variational.means, K_inv_sigma, variational.mixing_coefficients, centres, covs, K, title, savefigpath=filename, xylims=None)
        variational.log_likelihood = log_likelihood(X, variational.mixing_coefficients, variational.means, K_inv_sigma)
        variational.LL_error = true_log_likelihood - variational.log_likelihood
        
        # Save copy of class with all current params for plotting etc
        variational_memory.append(deepcopy(variational))
        samples = np.random.choice(N, size=minibatch_size, replace=False)
     
        # # EM steps, calculate ELBO
        t_start[i] = time.time()
        variational.E_step(X, samples)
        variational.M_step(X, joint, samples, t=i)
        t_end[i] = time.time()

    t_fin = time.time()
        
    # %% Saving animation, plotting ELBO/other key params
    
    "Make and display gif of mixture model over time" 
    gifname = make_gif(gifplots_dir, run_dir, gifname=update_type)
    # Empty plots directory for next run
    # for file in os.listdir(filedir):
    #   os.remove(os.path.join(filedir,file))
      
    "extracting from memory, and plotting, salient params"
    plt.close('all')
    ELBOs = np.array([variational_memory[n].ELBO for n in range(N_its)])
    LLs = np.array([variational_memory[n].LL_error for n in range(N_its)])
    alphas = np.array([variational_memory[n].alpha for n in range(N_its)])
    # Var_E_d_alphas = np.array([variational_memory[n].Var_E_d_alpha for n in range(1,N_its)])
    mixing_coefficients = np.array([variational_memory[n].mixing_coefficients for n in range(N_its)])
    varx = np.array([variational_memory[n].covariances[:,0,0] for n in range(N_its)])
    vary = np.array([variational_memory[n].covariances[:,1,1] for n in range(N_its)])
    covxy = np.array([variational_memory[n].covariances[:,1,0] for n in range(N_its)])
    
    plot_1D_param(ELBOs.reshape(-1,1), 'ELBO', 1, savefigdir=run_dir)
    plot_1D_param(LLs.reshape(-1,1), 'Log-likelihood error', 1, savefigdir=run_dir)
    plot_1D_param(alphas, 'alphas', K, savefigdir=run_dir)
    # plot_1D_param(Var_E_d_alphas, 'Variation of gradient estimates wrt alpha', K, savefigdir=run_dir)
    # plot_1D_param(mixing_coefficients, 'Mixing coefficients (E[pi_k])', K)
    # plot_K_covs(varx,vary,covxy,K)
    plt.show()
    
    # if update_type == 'GD':
    #     d_means_mag = np.array([variational_memory[n].means for n in range(1,N_its)])
    
    """Makes animation showing the ellipses for the distribution of 
    \mu_k ~ N{\mu_k|m_k, C_k} over time"""
    # plot_cov_ellipses_gif(variational_memory, N_its, K, gifname=' --- Mu distribution')
    
    """plot base step size"""
    rho_t = (gd_schedule(t=np.arange(N_its))['alpha'])
    plt.figure()
    plt.plot(rho_t)
    plt.xlabel('Iterations')
    plt.title('rho_t, base step size')
    plt.show()
    
    # %% SAVING STUFF
    
    run_info = {
        'true_log_likelihood' : true_log_likelihood,
        'final_ELBO' : variational.ELBO,
        't0': t0,
        't_start': t_start,
        't_end': t_end,
        't_fin': t_fin,
        'N': N,
        'seed': seed,
        'num_clusters': num_clusters,
        'N_its': N_its,
        'minibatch_size': minibatch_size,
        'update_type': update_type,
        'centres': centres,
        'covs': covs,
        'weights': weights,
        'joint': joint.__dict__,
        'gd_step_params': {scale:'scale', 'delay':delay, 'forgetting':forgetting},
        'step_sizes': step_sizes,
        'X': X,
        }
    
    with open(join(run_dir, 'run_info.txt'), 'w') as f:
        print(run_info, file=f)
        
    v_mem = []
    for v in variational_memory:
        v_n = v.__dict__
        _ = v_n.pop('gd_schedule')
        v_mem.append(v_n)
        
    run = {
           'run_info' : run_info, 
           'variational_memory' : variational_memory
           }
    
    pickle.dump(run, open(join(run_dir, "run_info.pkl"), "wb"))

