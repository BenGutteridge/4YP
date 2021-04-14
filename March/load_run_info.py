# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:15:27 2021

@author: benpg
"""

import pickle
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

run_dir = r"C:\Users\benpg\Documents\4YP\Running_experiments\2021-04-14 13;19;17 --- PW"
run = pickle.load(open(join(run_dir, 'run_info.pkl'), "rb"))

run_info = run['run_info']
v_mem = run['variational_memory']

N_its = run_info['N_its']

plt.figure()
ELBOs = np.array([v_mem[i].ELBO for i in range(1,N_its)])
plt.plot(ELBOs)
plt.show()