# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:09:34 2021

@author: benpg
"""

import numpy as np
from numpy.linalg import inv
from statistics_of_observed_data import N_k, x_k_bar, S_k
from CAVI_updates import alpha_k as update_alpha, m_invC_k as update_means_precisions
from calculate_responsibilities import r_nk
from grad_funcs import L_grad_alpha, L_grad_m, L_grad_invC
from calculate_ELBO import E_ln_p_X_given_Z_mu, E_ln_p_Z_given_pi, E_ln_p_pi, E_ln_p_mu
from calculate_ELBO import E_ln_q_Z, E_ln_q_pi, E_ln_q_mu

"""TODO: Have some sort of 'gd_schedule' arg - simple at first but can add more complexity
query it at each iteration
"""
class JointDistribution:
    def __init__(self, alpha, mean, covariance=None, precision=None):
        """

        :param alpha: Parameter of Dirichlet distribution for mixture weights
        :param mean:
        :param covariance:
        :param precision:
        """
        self.alpha = alpha
        self.mean = mean

        if covariance is not None:
            self.covariance = covariance
            self.precision = np.linalg.inv(covariance)
        elif precision is not None:
            self.precision = precision
            self.covariance = inv(precision)
        else:
            raise ValueError("Must specify one of precision and covariance.")


class VariationalDistribution:
    """
    Represents the variational distribution and its parameters. Contains
    methods for the updating of variational parameters via CAVI or GD updates,
    and the calculation of the ELBO.
    ...

    Attributes
    ----------
    update_type : str
        type of update scheme to use: GD, CAVI (TBA: SGD, BQ)
    responsibilities : (N,K) array
        variational parameter of latent variable Z, each row the calculated
        responsibility of the K components for each x_n
    alpha : (K,) array
        variational parameter of component weights pi
    means : (K,D) array
        variational mean parameters of normally distributed component means {mu_k}
    covariances : (K,D,D) array
        variational covariance parameters of N-distributed component means {mu_k}
    
    NK : (K,) array
        sum of responsibilities over N points for each of K components
    xbar : (K,D) array
        responsibility-weighted mean of dataset for each of K components
    SK : (K,D,D) array
        responsibility-weighted covariance of dataset for each of K components
        
    inv_sigma : (D,D) array
    D, K : ints
    
    Methods
    -------
    calculate_weighted_statistics()
        Calculates and sets attributes NK, xbar, SK
    M_step()
        Updates variational params alpha, means, covariances via CAVI or GD, 
        according to update_type attribute. 'Maximisation' step given responsibilities.
    E_step()
        'Expectation' step: Updates responsibilities according to current
        variational params
    """
    ALPHA_LB = 0.1

    def __init__(self, initial_responsibilities, inv_sigma, update_type,
                 step_sizes={'alpha': 1.0, 'm': 1e-2, 'invC': 1e-4}):
        self.update_type = update_type
        self.step_sizes = step_sizes

        self.responsibilities = initial_responsibilities

        self.K = initial_responsibilities.shape[1]
        self.D = 2

        self.inv_sigma = inv_sigma
        self.alpha = np.zeros(self.K)
        self.means = np.zeros((self.K, self.D))
        self.precisions = self.covariances = np.zeros((self.K, self.D, self.D))

        self.NK = np.zeros(self.K)
        self.xbar = np.zeros((self.K,self.D))
        self.SK = np.zeros((self.K,self.D,self.D))
        
    def calculate_weighted_statistics(self, X):
        """
        Calculates frequently used statistics of the responsibilities and the
        dataset and sets them as attributes of the distribution class. Roughly
        correspond to N, mean and covariance of X, weighted by responsibility.
        Reproduced from Eqns 10.51-10.53 in Bishop.
        
        NK : (K,) array
        sum of responsibilities over N points for each of K components
        xbar : (K,D) array
        responsibility-weighted mean of dataset for each of K components
        SK : (K,D,D) array
        responsibility-weighted covariance of dataset for each of K components
        """
        r = self.responsibilities
        
        for k in range(self.K):
            self.NK[k] = N_k(r[:,k])                                # Eqn 10.51
            self.xbar[k] = x_k_bar(self.NK[k], r[:,k], X)           # Eqn 10.52
            self.SK[k] = S_k(self.NK[k], r[:,k], X, self.xbar[k])   # Eqn 10.53

    def M_step(self, X, joint):
        if self.update_type == "CAVI":
            self._M_step_CAVI(X, joint)
        elif self.update_type == "GD":
            self._M_step_GD(X, joint)

    def _M_step_CAVI(self, X, joint):
        self.calculate_weighted_statistics(X)

        for k in range(self.K):
            # TODO: rename, get rid of _k everywhere
            self.alpha[k] = update_alpha(self.NK[k], joint.alpha)
            self.means[k], self.precisions[k] = \
                update_means_precisions(self.NK[k], self.xbar[k], joint.mean, joint.precision, self.inv_sigma)
            self.covariances[k] = inv(self.precisions[k])
            
    def _M_step_GD(self, X, joint):
        # performing GD update for M step rather than CAVI update equations
        self.calculate_weighted_statistics(X)

        # TODO: consider putting gradient step stuff in two separate methods, one for computing the updates and one for
        # applying the updates

        # Compute gradient updates
        # TODO: point to equation in Bishop or elsewhere whenever you've just reproduced something
        self.d_alpha = L_grad_alpha(joint.alpha, self.alpha, self.NK)
        self.d_m = L_grad_m(self.means, joint.mean, joint.precision, self.inv_sigma, self.NK, self.xbar)
        self.d_invC = L_grad_invC(self.means, self.precisions, self.inv_sigma, joint.precision, self.NK)

        # Apply gradient updates
        for k in range(self.K): 
            # constraints: alpha>0. Setting alpha>.1 as psi'(alpha->0) -> inf
            self.alpha[k] = np.max((self.ALPHA_LB, self.alpha[k] + (self.d_alpha[k]*self.step_sizes['alpha'])))
            self.means[k] = self.means[k] + self.d_m[k]*self.step_sizes['m']
            self.precisions[k] = self.precisions[k] + self.d_invC[k]*self.step_sizes['invC']
            self.covariances[k] = inv(self.precisions[k])

    # TODO: insert bishop ref for r_nk - maybe clean it up a little bit (later)? Looks opaque
    def E_step(self, X):
        N = X.shape[0]
        for k in range(self.K):
            if self.alpha[k] <= self.ALPHA_LB:
                self.responsibilities[:, k] = np.zeros(N) # fix?
            else:
                for n in range(N):
                    self.responsibilities[n, k] = r_nk(k, self.alpha, self.means, self.covariances, self.inv_sigma, X[n])
                    
    def E_step_minibatch(self, X, alpha_lb=0.1):
        # xn = X[np.random.randint(X.shape[0])]
        pass

    def perturb_variational_params(self, non_diag=False):
        # fill in
        pass
                    
    def update_precision_from_covariance(self):
        for k in range(self.K):
            self.precisions[k] = inv(self.covariances[k])
            
    def calculate_ELBO(self, joint):
        """Based on equation 10.70-10.77 from Bishop, with some modifications due to our fixed covariance."""
        alpha0 = joint.alpha        # prior param for Dirichlet distributed pi
        m0 = joint.mean             # prior mean param m0 for N-distributed mu 
        C0 = joint.covariance       # prior covariance param for N-distirbuted mu
        alpha = self.alpha          # updated ~posterior alpha
        m = self.means              # updated ~posterior m
        C = self.covariances        # updated ~posterior covariance (not in Bishop)
        r = self.responsibilities   # responsibilities
        NK = self.NK
        xbar = self.xbar
        SK = self.SK
        invSig = self.inv_sigma     # fixed precision for all Gaussians
    
        p1 = E_ln_p_X_given_Z_mu(m, invSig, C0, NK, xbar, SK, D=2)  # ~ Eqn 10.71
        p2 = E_ln_p_Z_given_pi(r, alpha)                            # = Eqn 10.72
        p3 = E_ln_p_pi(alpha0, alpha)                               # = Eqn 10.73
        p4 = E_ln_p_mu(m, C, m0, inv(C0), D=2)                      # ~ Eqn 10.74
        q1 = E_ln_q_Z(r)                                            # = Eqn 10.75
        q2 = E_ln_q_pi(alpha)                                       # = Eqn 10.76
        q3 = E_ln_q_mu(C, D=2)  
                                    # ~ Eqn 10.77
        self.ELBO = p1+p2+p3+p4-q1-q2-q3
            