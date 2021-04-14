# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:09:34 2021

@author: benpg
"""

import numpy as np
from numpy.linalg import inv
from copy import copy
from statistics_of_observed_data import N_k, x_k_bar, S_k
from CAVI_updates import alpha_k as update_alpha, m_invC_k as update_means_precisions
from calculate_responsibilities import calculate_ln_rho_nk
from grad_funcs import L_grad_alpha, L_grad_m, L_grad_C
from calculate_ELBO import E_ln_p_X_given_Z_mu, E_ln_p_Z_given_pi, E_ln_p_pi, E_ln_p_mu
from calculate_ELBO import E_ln_q_Z, E_ln_q_pi, E_ln_q_mu
from grad_est_utils import grad_H_q, sample_mu, sample_pi, evaluate_cost_SFE
from grad_est_utils import grad_ln_q_pi, grad_ln_q_mu
from grad_est_utils import grad_alpha_f, grad_m_f, grad_rootC_f
from grad_est_utils import sample_pathwise_mu, sample_pathwise_pi
from grad_est_utils import calculate_iwae_weights

np.random.seed(42)

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

    def __init__(self, initial_responsibilities, inv_sigma, update_type, gd_schedule):
        self.update_type = update_type

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
        
        self.gd_schedule = gd_schedule
        
    def initialise_params(self, init_alpha=10):
        """Initialise variational params before GD (or other) updates"""
        self.alpha = np.ones(self.K)*init_alpha
        self.means = 2*np.random.rand(self.K,self.D) - 1
        self.covariances = np.array([np.eye(2) for _ in range(self.K)])
        self.update_precisions_from_covariances()

    
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

    def M_step(self, X, joint, samples=None, t=None):
        """
        Undertakes M-step, optimising ELBO by updating variational params.
        Uses attribute 'update_type' to pass onto daughter M-step methods
        for different kinds of updates, CAVI, GD (minibatches TBA)
        """
        if self.update_type == "CAVI":
            self._M_step_CAVI(X, joint)
        elif self.update_type == "GD":
            self._M_step_GD(X, joint, t)
        elif self.update_type == "SGD":
            self._M_step_SGD(X, joint, samples, t)
        elif self.update_type == "SNGD":
            self._M_step_SNGD(X, joint, samples, t)
        elif self.update_type == "SFE":
            self._M_step_SFE(joint, X, t)
        elif self.update_type == "PW":
            self._M_step_pathwise(joint, X, t)

    def _M_step_CAVI(self, X, joint):
        """CAVI updates. Modified from Bishop Eqns 10.58-62"""
        self.calculate_weighted_statistics(X)

        for k in range(self.K):
            self.alpha[k] = update_alpha(self.NK[k], joint.alpha)   # = Eqn 10.58
            self.means[k], self.precisions[k] = \
                update_means_precisions(self.NK[k], self.xbar[k], joint.mean, joint.precision, self.inv_sigma)
            self.covariances[k] = inv(self.precisions[k])           # ~ Eqn 10.61
            
    def _M_step_SNGD(self, X, joint, samples, t):
        """
        stochastic natural gradient descent, using the fact that NGD is 
        equivalent to CAVI updates with a step size of 1
        
        """
        self.step_sizes = self.gd_schedule(t)
        N, S = X.shape[0], samples.shape[0]
        nat_grad_alpha = np.zeros((S, self.K))
        nat_grad_lam1 = np.zeros((S, self.K, self.D))
        nat_grad_lam2 = np.zeros((S, self.K, self.D, self.D))
        
        lam1 = np.array([np.dot(self.means[k], self.precisions[k]) for k in range(self.K)]) # (K,D)
        lam2 = np.array([-0.5*self.precisions[k] for k in range(self.K)])                  # (K,D,D)
        
        # Calculate natural gradients for each sample
        for i in range(samples.shape[0]):
            N_r_nk = np.array([N*self.responsibilities[samples[i],k] for k in range(self.K)])
            xbar = np.array([X[samples[i]] for _ in range(self.K)])
            
            nat_grad_alpha[i] = (update_alpha(N_r_nk, joint.alpha) - self.alpha)
            for k in range(self.K):
                nat_grad_lam1[i,k] = np.dot(joint.mean, joint.precision) + N_r_nk[k]*np.dot(xbar[k], self.inv_sigma) - lam1[k]
                nat_grad_lam2[i,k] = -0.5*(joint.precision + N_r_nk[k]*self.inv_sigma) - lam2[k]
        
        # Average natural gradient over minibatch
        nat_grad_alpha = np.mean(np.array(nat_grad_alpha), axis=0)  # (K,)
        nat_grad_lam1 = np.mean(np.array(nat_grad_lam1), axis=0)    # (K,D)
        nat_grad_lam2 = np.mean(np.array(nat_grad_lam2), axis=0)    # (K,D,D)
        
        # Perform updates, map natural lam params to variational means, covs
        self.alpha += self.step_sizes['alpha'] * nat_grad_alpha
        lam1 += self.step_sizes['lam1'] * nat_grad_lam1
        lam2 += self.step_sizes['lam2'] * nat_grad_lam2
        for k in range(self.K):
            self.means[k] = -0.5*np.dot(lam1[k], inv(lam2[k]))
            self.precisions[k] = -2*lam2[k]
            self.covariances[k] = inv(self.precisions[k])
        
    
    def _M_step_GD(self, X, joint, t):
        """
        Generate gradients of ELBO wrt variational params and take GD steps.
        """
        self.calculate_weighted_statistics(X)
        self.step_sizes = self.gd_schedule(t)  # TODO: sort this
        self._calculate_gradient_updates(joint)
        self._apply_gradient_updates()
        
    def _M_step_SGD(self, X, joint, samples, t):
        """
        Generate unbiased noisy estimates of gradient using a minibatching,
        each gradient calculated independently with an artificial dataset of
        minibatch point xi repeated N times. Gradients are averaged over the
        minibatch, weighted by the step size, and added to
        """
        N = X.shape[0]
        S = samples.shape[0]
        d_alpha, d_m, d_C = [], [], []
        self.step_sizes = self.gd_schedule(t)
        
        # Calculate SGD (single-sample) update steps for each sample in S
        for i in range(S):
            N_r_nk = np.array([N*self.responsibilities[samples[i],k] for k in range(self.K)])
            xbar = np.array([X[samples[i]] for _ in range(self.K)])
            d_alpha.append(L_grad_alpha(joint.alpha, self.alpha, N_r_nk))
            d_m.append(L_grad_m(self.means, joint.mean, joint.precision, self.inv_sigma, N_r_nk, xbar))
            d_C.append(L_grad_C(self.precisions, self.inv_sigma, joint.precision, N_r_nk))
        
        # Average SGD updates, weight by step size, and add to update variational params 
        d_alpha, d_m, d_C = np.array(d_alpha), np.array(d_m), np.array(d_C)
        E_d_alpha = np.mean(d_alpha, axis=0)
        E_d_m = np.mean(d_m, axis=0)
        E_d_C = np.mean(d_C, axis=0)
        self.Var_E_d_alpha = np.mean((d_alpha - E_d_alpha)**2, axis=0)
        self.Var_E_d_m = np.mean((d_m - E_d_m)**2, axis=0)
        self.Var_E_d_C = np.mean((d_C - E_d_C)**2, axis=0)
        self.alpha += self.step_sizes['alpha'] * E_d_alpha
        self.alpha = np.maximum(self.alpha, self.ALPHA_LB * np.ones(self.K)) # constraints: alpha>0. Setting alpha>.1 as psi'(alpha->0) -> inf
        self.means += self.step_sizes['m'] * E_d_m
        self.covariances += self.step_sizes['C'] * E_d_C
        self.update_precisions_from_covariances()
        
        
    def _M_step_SFE(self, joint, X, t, n_samples=10):
        """
        Score function estimator for GD optimisation of alpha, m, C
        Uses n_samples of pi, mu for MC gradient estimate
        """
        K, D = self.K, X.shape[1]
        self.step_sizes = self.gd_schedule(t)
        # using analytical entropy of q
        grad_H_alpha, grad_H_m, grad_H_C = grad_H_q(self)
        D_ELBO_alpha, D_ELBO_m, D_ELBO_C = np.zeros(K), np.zeros((K, D)), np.zeros((K,D,D))
        avg_f = float(evaluate_cost_SFE(self, joint, X, sample_pi(self), sample_mu(self)))/X.shape[0]
        for i in range(n_samples):
            mu_hat = sample_mu(self) # get a sample
            pi_hat = sample_pi(self)
            # 'black box' cost function
            f = float(evaluate_cost_SFE(self, joint, X, pi_hat, mu_hat))/X.shape[0]
            avg_f = np.mean((f, avg_f))
            print('f: ', f, 'avg_f: ', avg_f)
            # gradients of log measure for score function
            grad_alpha_ln_q = grad_ln_q_pi(self, pi_hat)
            grad_m_ln_q, grad_C_ln_q = grad_ln_q_mu(self, mu_hat)
            # collect gradients from multiple MC samples
            D_ELBO_alpha += (f - avg_f) * grad_alpha_ln_q
            D_ELBO_m += (f - avg_f) * grad_m_ln_q
            D_ELBO_C += (f - avg_f) * grad_C_ln_q
        # average over number of samples and set gradient for SGD update
        self.d_alpha = D_ELBO_alpha/n_samples + grad_H_alpha
        self.d_m = D_ELBO_m/n_samples + grad_H_m
        self.d_C = D_ELBO_C/n_samples + grad_H_C
        # use same gradient update method as GD
        self._apply_gradient_updates()
        
    def _M_step_pathwise(self, joint, X, t, n_samples=20, iwae=True):
        K, D = self.K, X.shape[1]
        self.step_sizes = self.gd_schedule(t)
        _grad_alpha_f = np.zeros((n_samples,K))
        _grad_m_f = np.zeros((n_samples, K, D))
        _grad_rootC_f = np.zeros((n_samples,K,D,D))
        pi_samples = np.zeros((n_samples,K)) 
        mu_samples = np.zeros((n_samples, K, D))
        for i in range(n_samples):
            # sample through paths
            mu_hat, xi_hat = sample_pathwise_mu(self) # get a sample
            pi_hat, z_hat, u_hat = sample_pathwise_pi(self)
            pi_samples[i], mu_samples[i] = pi_hat, mu_hat
            # evaluate grad f with variational params moved into f
            _grad_alpha_f[i] = grad_alpha_f(self, pi_hat, z_hat, u_hat)
            _grad_m_f[i] = grad_m_f(self, joint, X, mu_hat)
            _grad_rootC_f[i] = grad_rootC_f(self, joint, X, xi_hat, mu_hat)
        if iwae == True:
            self.weights = calculate_iwae_weights(n_samples, self, joint, 
                                                  X, pi_samples, mu_samples)
            _grad_alpha_f *= (n_samples * self.weights.reshape(-1,1))
            _grad_m_f *= (n_samples *self.weights.reshape(-1,1,1))
            _grad_rootC_f *= (n_samples * self.weights.reshape(-1,1,1,1))
        # average over number of samples
        self.d_alpha = np.mean(_grad_alpha_f, axis=0)
        self.d_m = np.mean(_grad_m_f, axis=0)
        self.d_C = np.zeros(K)
        self.d_rootC = np.mean(_grad_rootC_f, axis=0)
        self.Var_E_d_alpha = np.mean((self.d_alpha - _grad_alpha_f)**2,axis=0)
        self.Var_E_d_m = np.mean((self.d_m - _grad_m_f)**2,axis=0)
        self.Var_E_d_rootC = np.mean((self.d_rootC - _grad_rootC_f)**2,axis=0)
        # use same gradient update method as GD
        self._apply_gradient_updates()
        # need to update rootC separately
        root_C = self.covariances**0.5
        root_C += self.d_rootC * self.step_sizes['C']
        self.covariances = root_C**2
        self.update_precisions_from_covariances()
            
        
    def _calculate_gradient_updates(self, joint):
        """ 
        Compute gradient updates
        Adapted from Xie VI tutorial Eqns 76, 60/82/85 and 
        PRML solution manual Problem 10.18 
        """
        self.d_alpha = L_grad_alpha(joint.alpha, self.alpha, self.NK) # Xie Eqn 76
        self.d_m = L_grad_m(self.means, joint.mean, joint.precision, self.inv_sigma, self.NK, self.xbar)
        self.d_C = L_grad_C(self.precisions, self.inv_sigma, joint.precision, self.NK)

    def _apply_gradient_updates(self):
        """Applying gradient step updates using calculated gradients and
        step schedule from self.gd_schedule"""
        for k in range(self.K): 
            # constraints: alpha>0. Setting alpha>.1 as psi'(alpha->0) -> inf
            self.alpha[k] = np.max((self.ALPHA_LB, self.alpha[k] + (self.d_alpha[k]*self.step_sizes['alpha'])))
            self.means[k] = self.means[k] + self.d_m[k]*self.step_sizes['m']
            self.covariances[k] = self.covariances[k] + self.d_C[k]*self.step_sizes['C']
            self.precisions[k] = inv(self.covariances[k])

    
    def E_step(self, X, samples=None):
        """
        Performs E-step, optimising ELBO by updating variational param
        responsibility of latent variables Z.
        Uses Eqn 10.49 (and 10.46) from Bishop
        """
        if self.update_type in ['SGD','SNGD'
                                ]:
            self._E_step_minibatch(X, samples)
        else:
            self._E_step_batch(X)


    def _E_step_batch(self, X):
        """
        Update responsibilities of entire dataset
        """
        N = X.shape[0]
        self.calculate_responsibilities(N, X, samples=None)
        

    def _E_step_minibatch(self, X, samples):
        """
        Updates the responsibility only of the minibatch of points
        """
        N = X.shape[0]
        self.calculate_responsibilities(N, X, samples=samples)

                    
    def update_precisions_from_covariances(self):
        for k in range(self.K):
            self.precisions[k] = inv(self.covariances[k])
        
    def calculate_mixing_coefficients(self):
        # E[pi_k], Bishop Eqn B.17
        self.mixing_coefficients = self.alpha/np.sum(self.alpha)
        
    def calculate_responsibilities(self, N, X, samples=None):
        if samples is None: samples = np.arange(N)
        rho, r = np.zeros((N, self.K)), np.zeros((N, self.K))
        # r = self.responsibilities
        # 1. calculate all rho_nk (from chosen samples)
        for n in range(N):
            if n in samples:
                for k in range(self.K):
                    if self.alpha[k] >= self.ALPHA_LB:
                        rho[n][k] = np.exp(calculate_ln_rho_nk(k,self.alpha,self.means,self.covariances,self.inv_sigma,X[n],D=2))
        # 2. use rho_nk to calculate r_nk
        live_components = np.arange(self.K)[self.alpha > self.ALPHA_LB]
        for n in samples:
            for k in live_components:
                try:
                    r[n][k] = float(rho[n][k]) / float(np.sum(rho[n][:]))
                    assert not np.isnan(r[n][k])
                except AssertionError:
                    print('NaN in r[%d][%d], rho[n][:] = ' % (n,k), rho[n][:])
                except ZeroDivisionError:
                    # when all components bear negligible responsibility
                    r[n][k] = 0.
        self.responsibilities = r # n.b. resets all previous responsibilities
            
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
            