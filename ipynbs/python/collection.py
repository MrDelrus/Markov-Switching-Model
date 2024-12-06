import yfinance as yf
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from python.model import MarkovSwitchingModel, loglikelyhood
from numpy.lib.stride_tricks import sliding_window_view

# Fitting pipeline
def fit_msm(returns, window_shape=250, *, position=0):
  model = MarkovSwitchingModel(loglikelyhood)
  minimums = []
  for window in tqdm(sliding_window_view(returns[:-1], window_shape=window_shape), position=position, leave=False):
    model.fit(window)
    params = model.get_params()
    minimums.append([params[0][0], params[0][1], params[0][2], params[0][3], params[0][4], params[0][5], params[1], params[2]])
  return pd.DataFrame(minimums, columns=['a1', 'sigma1', 'a2', 'sigma2', 'p12', 'p21', 'p1_final', 'p2_final'])

# Preprocessing
def preprocess_data(ticker_name, start='2022-01-01', end='2023-12-31'):
  ticker = yf.Ticker(ticker_name)
  prices = ticker.history(start=start, end=end, interval='1d')[['Close']].reset_index()
  return np.array([ prices["Close"][T + 1] / prices["Close"][T] - 1 for T in range(0, len(prices["Close"]) - 1) ]), prices["Date"][1:]

# Fitting on ticker
def fit_on_ticker(ticker_name, output_path, *, window_shape=250, start='2022-01-01', end='2023-12-31', position=0):
  returns, dates = preprocess_data(ticker_name, start, end)
  fit_msm(returns, window_shape, position=position).set_index(dates[window_shape:]).to_csv(output_path)