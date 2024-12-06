import numpy as np
import pandas as pd
import scipy.stats as sps

from python.model import MarkovSwitchingModel, loglikelyhood
from python.em import estimate_var_em
from python.collection import preprocess_data
from numpy.lib.stride_tricks import sliding_window_view

class Testing:
  def __init__(self, vars, quant, returns, stocks, times, window):
    self.vars = vars
    self.quant = quant
    self.returns = np.array(returns)
    self.stocks = np.array(stocks)
    self.times = times
    self.window = window

  def testing(self):
    returns_stocks = np.array(((self.returns.T @ self.stocks.T)[self.window:]).T)
    exceptions = np.where(returns_stocks < self.vars, 1, 0)
    exception_sum = np.sum(exceptions, axis=1)
    # bin test
    p_values = np.array([sps.binomtest(exception_sum_stock, n=self.times - self.window, p=self.quant).pvalue for exception_sum_stock in exception_sum])
    bin_passed = np.where(p_values > 0.05, 1, 0)
    #independence test
    exceptions_str = np.where(returns_stocks < self.vars, "1", "0")
    t01 = np.array([''.join(exception).count("01") for exception in exceptions_str])
    t11 = np.array([''.join(exception).count("11") for exception in exceptions_str])
    t10 = np.array([''.join(exception).count("10") for exception in exceptions_str])
    t00 = np.array([''.join(exception).count("00") for exception in exceptions_str])
    #other independence test
    n = self.times - self.window
    independence_st = n * (np.abs(t00 * t11 - t01*t10) - n / 2)**2 / (t00 + t01) / (t00 + t10) / (t11 + t01) / (t11 + t10)
    indepen_passed = np.where(independence_st  < sps.chi2.ppf(0.95, df=1), 1, 0)

    return bin_passed, indepen_passed


def test(vars, returns, alpha, window_shape) :

  return Testing(np.expand_dims(vars, axis=0), alpha, np.expand_dims(returns, axis=0), np.array([1]).reshape(-1, 1), len(returns), window_shape).testing()

# Backtesting function
def make_backtest(ticker_name, alpha, *, start='2010-01-01', end='2020-01-01', window_shape=250):

  returns = preprocess_data(ticker_name, start=start, end=end)[0]

  vars_old = []
  for window in sliding_window_view(returns[:-1], window_shape=window_shape):
    vars_old.append(estimate_var_em(window, alpha))

  em1, em2 = test(vars_old, returns, alpha, window_shape)

  returns_new = returns[window_shape:]
  vars_new = np.zeros(returns_new.shape[0])
  model = MarkovSwitchingModel(loglikelyhood)
  params_df = pd.read_csv('data/' + ticker_name + '.csv')

  exceptions = np.where(vars_new < returns_new)

  for i in range(vars_new.shape[0]):
    row = params_df.iloc[i]
    minimum = np.array([row['a1'], row['sigma1'], row['a2'], row['sigma2'], row['p12'], row['p21']])
    p1 = row['p1_final']
    p2 = row['p2_final']
    model.set_params(minimum, p1, p2)
    vars_new[i] = model.predict(alpha)

  msm1, msm2 = test(vars_new, returns, alpha, window_shape)

  return msm1[0], msm2[0], em1[0], em2[0], exceptions