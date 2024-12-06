import numpy as np
import scipy.stats as sps

from python.lib import binary_search
from scipy.optimize import Bounds, minimize
from numba import jit

# Model
class MarkovSwitchingModel:
  def __init__(self, loglikelyhood, *, epsilon=1e-6, starting_probs=0.3, solver=binary_search):
    """
    _minimum is [a1, sigma1, a2, sigma2, p12, p21]
    """
    self._loglikelyhood = loglikelyhood
    self._epsilon = epsilon
    self._starting_probs = starting_probs
    self._solver = solver
    self._minimum = None
    self._p1 = None
    self._p2 = None

  def set_params(self, minimum, p1, p2):
    self._minimum = minimum
    self._p1 = p1
    self._p2 = p2

  def get_params(self):
    return self._minimum, self._p1, self._p2

  def fit(self, returns):
    a0 = returns.mean()
    sigma0 = returns.std()

    if self._minimum is None:
      self._minimum = np.array([a0, sigma0 / 2, a0, 2 * sigma0, self._starting_probs, self._starting_probs])

    bounds = Bounds(
      [-a0 - 3 * sigma0, sigma0 / 3, -a0 - 3 * sigma0, sigma0, 0, 0],
      [a0 + 3 * sigma0, sigma0, a0 + 3 * sigma0, 3 * sigma0, 1, 1]
    )

    self._minimum = minimize(
      lambda x : self._loglikelyhood(x, returns)[0],
      self._minimum,
      method='powell',
      tol=self._epsilon,
      bounds=bounds,
      options={'maxiter' : 50}
    ).x

    _, self._p1, self._p2 = self._loglikelyhood(self._minimum, returns)

    return self

  def predict(self, var_alpha):

    a1 = self._minimum[0]
    sigma1 = self._minimum[1]
    a2 = self._minimum[2]
    sigma2 = self._minimum[3]
    p12 = self._minimum[4]
    p21 = self._minimum[5]

    p11 = 1 - p12
    p22 = 1 - p21

    def get_prob(x):
      p1 = self._p1 * p11 + self._p2 * p21
      p2 = self._p1 * p12 + self._p2 * p22
      return p1 * sps.norm(loc=a1, scale=sigma1).cdf(x) + p2 * sps.norm(loc=a2, scale=sigma2).cdf(x)

    return self._solver(lambda x : get_prob(x) - var_alpha, -10, 10)

@jit(nopython=False)
def loglikelyhood(x : np.ndarray[np.float64], returns : np.ndarray[np.float64]):
  a1 = x[0]
  sigma1 = x[1]
  a2 = x[2]
  sigma2 = x[3]
  p12 = x[4]
  p21 = x[5]
  p11 = 1 - p12
  p22 = 1 - p21
  phi1 = lambda x : 1 / np.sqrt(2 * np.pi * sigma1 * sigma1) * np.exp(- ( (x - a1) / sigma1 ) ** 2 / 2)
  phi2 = lambda x : 1 / np.sqrt(2 * np.pi * sigma2 * sigma2) * np.exp(- ( (x - a2) / sigma2 ) ** 2 / 2)
  n = returns.shape[0]

  rho = np.ones(n + 1)

  p1 = np.zeros(n + 1)
  p2 = np.zeros(n + 1)
  p1[0] = p21 / (p12 + p21)
  p2[0] = p12 / (p12 + p21)
  for t in np.arange(n):
    p1_next = p1[t] * p11 + p2[t] * p21
    p2_next = p1[t] * p12 + p2[t] * p22
    rho[t + 1] = p1_next * phi1(returns[t]) + p2_next * phi2(returns[t])
    p1[t + 1] = phi1(returns[t]) * p1_next / rho[t + 1]
    p2[t + 1] = phi2(returns[t]) * p2_next / rho[t + 1]

  rho = np.log(rho)
  return -rho.sum(), p1[n], p2[n]