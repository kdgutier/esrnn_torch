import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd
from math import sqrt


########################
# UTILITY MODELS
########################

def detrend(insample_data):
  """
  Calculates a & b parameters of LRL
  :param insample_data:
  :return:
  """
  x = np.arange(len(insample_data))
  a, b = np.polyfit(x, insample_data, 1)
  return a, b

def deseasonalize(original_ts, ppy):
  """
  Calculates and returns seasonal indices
  :param original_ts: original data
  :param ppy: periods per year
  :return:
  """
  """
  # === get in-sample data
  original_ts = original_ts[:-out_of_sample]
  """
  if seasonality_test(original_ts, ppy):
    # ==== get moving averages
    ma_ts = moving_averages(original_ts, ppy)

    # ==== get seasonality indices
    le_ts = original_ts * 100 / ma_ts
    le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
    le_ts = np.reshape(le_ts, (-1, ppy))
    si = np.nanmean(le_ts, 0)
    norm = np.sum(si) / (ppy * 100)
    si = si / norm
  else:
    si = np.ones(ppy)

  return si

def moving_averages(ts_init, window):
  """
  Calculates the moving averages for a given TS
  :param ts_init: the original time series
  :param window: window length
  :return: moving averages ts
  """
  """
  As noted by Professor Isidro Lloret Galiana:
  line 82:
  if len(ts_init) % 2 == 0:
  
  should be changed to
  if window % 2 == 0:
  
  This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
  In order for the results to be fully replicable this change is not incorporated into the code below
  """
  ts_init = pd.Series(ts_init)
  
  if len(ts_init) % 2 == 0:
    ts_ma = ts_init.rolling(window, center=True).mean()
    ts_ma = ts_ma.rolling(2, center=True).mean()
    ts_ma = np.roll(ts_ma, -1)
  else:
    ts_ma = ts_init.rolling(window, center=True).mean()

  return ts_ma

def seasonality_test(original_ts, ppy):
  """
  Seasonality test
  :param original_ts: time series
  :param ppy: periods per year
  :return: boolean value: whether the TS is seasonal
  """
  s = acf(original_ts, 1)
  for i in range(2, ppy):
    s = s + (acf(original_ts, i) ** 2)

  limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

  return (abs(acf(original_ts, ppy))) > limit

def acf(data, k):
  """
  Autocorrelation function
  :param data: time series
  :param k: lag
  :return:
  """
  m = np.mean(data)
  s1 = 0
  for i in range(k, len(data)):
    s1 = s1 + ((data[i] - m) * (data[i - k] - m))

  s2 = 0
  for i in range(0, len(data)):
    s2 = s2 + ((data[i] - m) ** 2)

  return float(s1 / s2)

class Naive:
  """
  Naive model.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init):
    """
    ts_init: the original time series
    ts_naive: last observations of time series
    """
    self.ts_naive = [ts_init[-1]]
    return self

  def predict(self, h):
    return np.array(self.ts_naive * h)
    
class SeasonalNaive:
  """
  Seasonal Naive model.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init, seasonality):
    """
    ts_init: the original time series
    frcy: frequency of the time series
    ts_naive: last observations of time series
    """
    self.ts_seasonal_naive = ts_init[-seasonality:]
    return self

  def predict(self, h):
    repetitions = int(np.ceil(h/len(self.ts_seasonal_naive)))
    y_hat = np.tile(self.ts_seasonal_naive, reps=repetitions)[:h]
    return y_hat

class Naive2:
  """
  Naive2: Naive after deseasonalization.
  """
  def __init__(self, seasonality):
    self.seasonality = seasonality
    
  def fit(self, ts_init):
    seasonality_in = deseasonalize(ts_init, ppy=self.seasonality)
    windows = int(np.ceil(len(ts_init) / self.seasonality))
    
    self.ts_init = ts_init
    self.s_hat = np.tile(seasonality_in, reps=windows)[:len(ts_init)]
    self.ts_des = ts_init / self.s_hat
            
    return self
    
  def predict(self, h):
    s_hat = SeasonalNaive().fit(self.s_hat,
                                seasonality=self.seasonality).predict(h)
    r_hat = Naive().fit(self.ts_des).predict(h)        
    y_hat = s_hat * r_hat
    return y_hat


class RandomWalkDrift:
  """
  RandomWalkDrift: Random Walk with drift.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init):
    self.drift = (ts_init[-1] - ts_init[0])/(len(ts_init)-1)
    self.naive = [ts_init[-1]]
    return self

  def predict(self, h):
    naive = np.array(self.ts_naive * h)
    drift = self.drift*np.array(range(1,h+1))
    y_hat = naive + drift
    return y_hat

########################
# METRICS
########################

def mse(y, y_hat):
  """
  Calculates Mean Squared Error.
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  return: MSE
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  mse = np.mean(np.square(y - y_hat)).item()
  return mse

def mape(y, y_hat):
  """
  Calculates Mean Absolute Percentage Error.
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  return: MAPE
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  mape = np.mean(np.abs(y - y_hat) / np.abs(y))
  return mape

def smape(y, y_hat):
  """
  Calculates Symmetric Mean Absolute Percentage Error.
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  return: sMAPE
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
  return smape

def mase(y, y_hat, y_train, seasonality):
  """
  Calculates Mean Absolute Scaled Error.
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  y_train: numpy array
    actual train values for Naive1 predictions
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  return: MASE
  """
  y_hat_naive = []
  for i in range(seasonality, len(y_train)):
      y_hat_naive.append(y_train[(i - seasonality)])

  masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
  mase = np.mean(abs(y - y_hat)) / masep
  return mase

########################
# PANEL EVALUATION
########################

def evaluate_panel(y_panel, y_hat_panel, metric,
                   y_insample=None, seasonality=None):
  """
  Calculates metric for y_panel and y_hat_panel
  y_panel: pandas df
    panel with columns unique_id, ds, y
  y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  return: list of metric evaluations
  """
  metric_name = metric.__code__.co_name

  y_panel = y_panel.sort_values(['unique_id', 'ds'])
  y_hat_panel = y_hat_panel.sort_values(['unique_id', 'ds'])
  if y_insample is not None:
      y_insample = y_insample.sort_values(['unique_id', 'ds'])

  assert len(y_panel)==len(y_hat_panel)
  assert all(y_panel.unique_id.unique() == y_hat_panel.unique_id.unique()), "not same u_ids"

  evaluation = []
  for u_id in y_panel.unique_id.unique():
    top_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'left'))
    bottom_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'right'))
    y_id = y_panel[top_row:bottom_row].y.to_numpy()

    top_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'left'))
    bottom_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'right'))
    y_hat_id = y_hat_panel[top_row:bottom_row].y_hat.to_numpy()
    assert len(y_id)==len(y_hat_id)

    if metric_name == 'mase':
      assert (y_insample is not None) and (seasonality is not None)
      top_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'left'))
      bottom_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'right'))
      y_insample_id = y_insample[top_row:bottom_row].y.to_numpy()
      evaluation_id = metric(y_id, y_hat_id, y_insample_id, seasonality)
    else:
      evaluation_id = metric(y_id, y_hat_id)
    evaluation.append(evaluation_id)
  return evaluation

def owa(y_panel, y_hat_panel, y_naive2_panel, y_insample, seasonality):
  """
  Calculates MASE, sMAPE for Naive2 and current model
  then calculatess Overall Weighted Average.
  y_panel: pandas df
    panel with columns unique_id, ds, y
  y_hat_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  return: OWA
  """
  total_mase = evaluate_panel(y_panel, y_hat_panel, mase, 
                              y_insample, seasonality)
  total_mase_naive2 = evaluate_panel(y_panel, y_naive2_panel, mase,
                                     y_insample, seasonality)
  total_smape = evaluate_panel(y_panel, y_hat_panel, smape)
  total_smape_naive2 = evaluate_panel(y_panel, y_naive2_panel, smape)

  assert len(total_mase) == len(total_mase_naive2)
  assert len(total_smape) == len(total_smape_naive2)
  assert len(total_mase) == len(total_smape)
  
  naive2_mase = np.mean(total_mase_naive2)
  naive2_smape = np.mean(total_smape_naive2) * 100

  model_mase = np.mean(total_mase)
  model_smape = np.mean(total_smape) * 100
  
  model_owa = ((model_mase/naive2_mase) + (model_smape/naive2_smape))/2
  return model_owa, model_mase, model_smape

def evaluate_prediction_owa(y_hat_df, y_train_df, X_test_df, y_test_df, 
                            naive2_seasonality):
    """
    y_hat_df: pandas df
      panel with columns unique_id, ds, y_hat
    y_train_df: pandas df
      panel with columns unique_id, ds, y
    X_test_df: pandas df
      panel with columns unique_id, ds, x
    y_test_df: pandas df
      panel with columns unique_id, ds, y, y_hat_naive2
    naive2_seasonality: int
      seasonality for the Naive2 predictions (needed for owa)
    model: python class
      python class with predict method
    """
    y_panel = y_test_df.filter(['unique_id', 'ds', 'y'])
    y_naive2_panel = y_test_df.filter(['unique_id', 'ds', 'y_hat_naive2'])
    y_naive2_panel.rename(columns={'y_hat_naive2': 'y_hat'}, inplace=True)
    y_hat_panel = y_hat_df
    y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

    model_owa, model_mase, model_smape = owa(y_panel, y_hat_panel, 
                                             y_naive2_panel, y_insample,
                                             seasonality=naive2_seasonality)

    print(15*'=', ' Model evaluation ', 14*'=')
    print('OWA: {} '.format(np.round(model_owa, 3)))
    print('SMAPE: {} '.format(np.round(model_smape, 3)))
    print('MASE: {} '.format(np.round(model_mase, 3)))
    return model_owa, model_mase, model_smape
