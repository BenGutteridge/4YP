# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:15:27 2021

@author: benpg
"""

import pickle
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
from tqdm import tqdm


def get_EM_times(t0, t_start, t_end):
    t_start, t_end = t_start - t0, t_end - t0
    lengths = t_end - t_start
    newtimes = np.zeros(len(lengths) + 1)
    for i in range(len(lengths)):
        newtimes[i+1] = lengths[i] + np.sum(lengths[:i])
    return newtimes
    
    
def plot_time(dir_of_dirs, initial_LL_error, legend=None):
    run_dirs = []
    for folder in os.listdir(dir_of_dirs):
        print(folder)
        run_dirs.append(folder)
    
    runs = []
    for run_dir in run_dirs:
        run = pickle.load(open(join(dir_of_dirs, run_dir, 'run_info.pkl'), "rb"))
        runs.append(run)
    import matplotlib 
    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 
    matplotlib.rcParams.update({'font.size': 12})
    LL_errors = [], []
    i = 0
    legends = []
    plt.figure(figsize=(8,6))
    plt.suptitle('Log-likelihood error')
    for run in runs:
        v_mem = run['variational_memory']
        run_info = run['run_info']
        ELBOs = np.array([initial_LL_error] + [v.ELBO for v in v_mem])
        LL_errors = np.array([v.LL_error for v in v_mem])
        # get times
        t0 = run_info['t0']
        t_start = run_info['t_start']
        t_end = run_info['t_end']
        t_fin = run_info['t_fin']
        newtimes = get_EM_times(t0, t_start, t_end)[:-1]
        legends.append("%s -- %d -- %s" % (run_info['update_type'], i, run_dirs[i]))
        if len(newtimes) > len(LL_errors):
            newtimes = newtimes[:len(LL_errors)]
        i += 1
        
        # # just one
        # plt.subplot(122)
        # plt.plot(LL_errors)
        # plt.xlabel('Iterations')
        # plt.ylim([50,500])
        # plt.subplot(121)
        # plt.plot(LL_errors)
        # plt.xlabel('Iterations')

        
        # # *** 1 x 2 ***
        # plt.subplot(121)
        # plt.plot(LL_errors)
        # plt.xlabel('Iterations')
        # # plt.ylim([0,20000])
        
        # plt.subplot(122)
        # plt.plot(newtimes, LL_errors)
        # # plt.setp(plt.subplot(122), yticks=[])
        # # plt.xlim([0,150])
        # # plt.ylim([0,20000])
        # plt.xlabel('Seconds')
        
        # *** 2 x 2 SNGD ***
        plt.subplot(221)
        plt.ylim([0,1000])
        plt.setp(plt.subplot(221), xticks=[])
        plt.plot(LL_errors)
           
        plt.subplot(222)
        plt.xlim([0,30])
        plt.ylim([0,1000])
        plt.setp(plt.subplot(222), xticks=[], yticks=[])
        plt.plot(newtimes, LL_errors)
           
        plt.subplot(223)
        plt.plot(LL_errors)
        plt.xlabel('Iterations')
        # plt.ylim([0,30000])
           
        plt.subplot(224)
        plt.plot(newtimes, LL_errors)
        plt.setp(plt.subplot(224), yticks=[])
        plt.xlim([-1,30])
        #plt.ylim([0,30000])
        plt.xlabel('Seconds')
           

    if legend is not None:
        # plt.figlegend(legend, loc='center')
        plt.legend(legend)
    else:
        plt.legend(legends)
    plt.tight_layout()
    plt.show()
    
from calculate_ELBO import calculate_ELBO_variance
def plot_ELBO_error_bars(dir_of_dirs, legend=None):
    run_dirs = []
    for folder in os.listdir(dir_of_dirs):
        run_dirs.append(folder)
        if len(run_dirs) == 4: break
    
    runs = []
    for run_dir in run_dirs:
        run = pickle.load(open(join(dir_of_dirs, run_dir, 'run_info.pkl'), "rb"))
        runs.append(run)
    
    i = 0
    legends = []
    plt.figure()
    for run in tqdm(runs):
        v_mem = run['variational_memory']
        run_info = run['run_info']
        minibatch_size, N = run_info['minibatch_size'], run_info['N']
        ELBOs = np.array([v.ELBO for v in v_mem])
        ELBOs *= (N/minibatch_size)  # correcting for error
        ELBO_expectations, ELBO_variances = get_variances_for_run(v_mem, run_info, run_info['X'])
        twosigs = 2*(ELBO_variances)**0.5
        ELBOs, twosigs = ELBOs[1:], twosigs[1:]
        legends.append("%s -- %d -- %s" % (run_info['update_type'], i, run_dirs[i]))
        i += 1
        plt.plot(ELBOs)
        plt.fill_between(np.arange(len(v_mem)-1), ELBOs + twosigs, ELBOs - twosigs, alpha=0.2)
        plt.title('ELBO')
        plt.xlabel('Its')
    plt.legend(legends)
    plt.show()
    
def get_variances_for_run(v_mem, run_info, X):
    alpha0 = run_info['joint']['alpha']
    m0 = run_info['joint']['mean']
    C0 = run_info['joint']['covariance']
    variances, expectations, i = np.zeros(len(v_mem)), np.zeros(len(v_mem)), 0
    for i in tqdm(range(1,len(v_mem))):
        expectations[i], variances[i] = calculate_ELBO_variance(v_mem[i], alpha0, m0, C0, 
                                               v_mem[i].samples, X)
    return expectations, variances
    
    
legend = ['Minibatch size = 1',
 'Minibatch size = 3',
 'Minibatch size = 5',
 'Minibatch size = 10',
 'Minibatch size = 20',
 'Minibatch size = 30',
 'Minibatch size = 50',
 'Minibatch size = 100',
 'Minibatch size = 200']


# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\SNGD_nsamples_runs\new_no_ELBO"
# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\SGD_n_samples_runs"
# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\PW_nsamples_runs"
# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\SNGD_nsamples_runs\new"

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\PW_nsamples_runs"
# legend = [1,1,1,1,3,5,10,20,100]

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\PW_nsamples_runs\1 3 5 10 20 100"
# legend = [1,3,5,10,20,100]

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\GD_step_size_runs" 
# legend = [1,3,5,10]

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\SGD_n_samples_runs\new_100"
# legend = [1,5,10,35,100]

# L = [1,3,5,10,20,30,50,100,200]
# # legend = ['L = %d' %l for l in L]
# legend = L

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\combined_SGD_SNGD\subset"
# legend = ['%d SNGD' % l for l in [1, 5, 10, 50]] + ['%d SGD' % l for l in [1, 5, 10, 50]]

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\combined_SGD_SNGD\subset\just 10"
# legend =['SNGD', 'SGD']

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\SNGD_nsamples_runs\new_100"
dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\sngd_smallersteps"
legend = [1,5,10,35,100]

# for PW, IWAE. lehend is number of samples
# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\PW_IWAE_n_samples"
# legend = [1, 5, 10, 20, 100]

# dir_of_dirs = r"C:\Users\benpg\Documents\4YP\Running_experiments\RUNS TO USE"
# legend = ['CAVI', 'PW, L=10', 'GD, scale=5', 'SGD, S=10', 'SNGD, S=35', ]

initial_LL_error = 8189.840160911926

plot_time(dir_of_dirs, initial_LL_error=initial_LL_error, legend=legend)
# plot_time(dir_of_dirs, initial_LL_error=initial_LL_error, legend=[1,3,5,10,20])
# plot_ELBO_error_bars(dir_of_dirs)

# run_dir = r"C:\Users\benpg\Documents\4YP\Running_experiments\2021-04-14 13;19;17 --- PW"
# run = pickle.load(open(join(run_dir, 'run_info.pkl'), "rb"))

# run_info = run['run_info']
# v_mem = run['variational_memory']

# N_its = run_info['N_its']

# plt.figure()
# ELBOs = np.array([v_mem[i].ELBO for i in range(1,N_its)])
# plt.plot(ELBOs)
# plt.show()