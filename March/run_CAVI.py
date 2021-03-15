import numpy as np
from statistics_of_observed_data import N_k, x_k_bar, S_k


class JointDistribution:
    def __init__(self, alpha, means, covariance=None, precision=None):
        """

        :param alpha: Parameter of Dirichlet distribution for mixture weights
        :param means:
        :param covariance:
        :param precision:
        """
        self.alpha = alpha
        self.means = means

        if covariance is not None:
            self.covariance = covariance
            self.precision = np.linalg.inv(covariance)
        elif precision is not None:
            self.precision = precision
            self.covariance = np.linalg.inv(precision)
        else:
            raise ValueError("Must specify one of precision and covariance.")


class VariationalDistribution:
    def __init__(self, initial_responsibilities):
        self.initial_responsibilities = initial_responsibilities
        self.K = initial_responsibilities.shape[1]
        


alpha_0 = 1e-3
m_0 = np.zeros(2)
C_0 = np.eye(2)
K = 2
N = 500


# Fixed covariance of every component of the Gaussian mixture
sigma = inv_sigma = np.eye(2)

initial_responsibilities = np.random.dirichlet(np.ones(K), N)  # Size (N, K)

variational_distribution = VariationalDistribution(initial_responsibilities)
joint = JointDistribution(alpha_0, m_0, covariance=C_0)


def M_step(X, variational_distribution, joint_distribution):
  r = variational_distribution.initial_responsibilities
  NK, xbar, SK = [],[],[]
  alpha = np.empty(K)
  m, invC = [np.zeros(2) for _ in range(K)], [np.zeros((2,2)) for _ in range(K)]

  for k in range(K):
    Nk = N_k(r[:,k])
    xkbar = x_k_bar(Nk, r[:,k], X)
    Sk = S_k(Nk, r[:,k], X, xkbar)

    joint_distribution.alpha[k] = alpha_k(Nk, alpha0)
    joint_distribution.means[k], joint_distribution.precision[k] = m_invC_k(Nk, xkbar, m0, invC0, invSig)

    NK.append(Nk)
    xbar.append(xkbar.reshape(2,))
    SK.append(Sk)

    # print('k=%d\nalpha'%k, alpha[k], '\nbeta', beta[k], '\nm', m[k], '\nW', W[k], '\nnu', nu[k])
  return alpha, m, inv(invC), NK, xbar, SK

