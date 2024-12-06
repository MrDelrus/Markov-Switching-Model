import scipy.stats as sps
import numpy as np

from sklearn.mixture import GaussianMixture
from python.lib import binary_search


def estimate_var_em(returns, alpha, n_components=2):
  gm = GaussianMixture(n_components=n_components).fit(returns.reshape(-1, 1))

  def mixture_cdf(x):
    res = 0;
    for i in range(n_components):
      res += gm.weights_[i] * sps.norm(loc=gm.means_[i][0], scale=np.sqrt(gm.covariances_[i][0][0])).cdf(x)
    return res

  return binary_search(lambda x : mixture_cdf(x) - alpha, -1, 1)
