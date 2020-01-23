import os
import numpy as np

import torch
import torch.nn as nn




class ES(nn.Module):
  def __init__(self, mc):
    super(ES, self).__init__()
    self.mc = mc
    self.max_num_series = self.mc.max_num_series
    self.seasonality = self.mc.seasonality
    self.output_size = self.mc.output_size

    # Level and Seasonality Smoothing parameters
    init_lev_sms = torch.ones((self.max_num_series, 1), dtype=torch.float64) * 0.5
    init_seas_sms = torch.ones((self.max_num_series, 1), dtype=torch.float64) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.seas_sms = nn.Parameter(data=init_seas_sms, requires_grad=True)

    init_seas = torch.ones((self.max_num_series, self.seasonality), dtype=torch.float64) * 0.5
    self.init_seas = nn.Parameter(data=init_seas, requires_grad=True)
    self.logistic = nn.Sigmoid()

  def compute_levels_seasons(self, ts_objects):
    # Parse ts_objects
    idxs = ts_objects.idxs
    y = ts_objects.y
    n_series, n_time = y.shape
    assert n_series == len(idxs)

    # Lookup embeddings
    lev_sms = self.logistic(self.lev_sms[idxs])
    seas_sms = self.logistic(self.lev_sms[idxs])
    init_seas = torch.exp(self.init_seas[idxs, :])

    # Initialize seasonalities, levels and log_diff_of_levels
    seasonalities = torch.zeros((n_series, self.seasonality+n_time), dtype=torch.float64)
    levels = torch.zeros((n_series, n_time), dtype=torch.float64)
    log_diff_of_levels = torch.zeros((n_series, n_time-1), dtype=torch.float64)

    seasonalities[:, :self.seasonality+1] = torch.cat((init_seas, init_seas[:,[0]]), 1)
    levels[:, 0] = y[:, 0] / seasonalities[:, 0]

    for t in range(1, n_time):
      newlev = lev_sms * (y[:, [t]] / seasonalities[:, [t]]) + (1-lev_sms) * levels[:, [t-1]]
      newseason = seas_sms * (y[:, [t]] / newlev) + (1-seas_sms) * seasonalities[:, [t]]
      log_diff = torch.log(newlev)-torch.log(levels[:, [t-1]])

      seasonalities[:, [t+self.seasonality]] += newseason
      levels[:, [t]] += newlev
      log_diff_of_levels[:, [t-1]] += log_diff

    # Completion of seasonalities if prediction horizon is larger than seasonality
    # Naive2 like prediction, to avoid recursive forecasting
    if self.output_size > self.seasonality:
      start_seasonality_ext = len(seasonalities) - self.seasonality
      end_seasonality_ext = start_seasonality_ext + self.output_size - self.seasonality
      seasonalities = torch.cat((seasonalities,
                                 seasonalities[:, start_seasonality_ext:end_seasonality_ext]), 1)

    return levels, seasonalities, log_diff_of_levels

  def forward(self, ts_objects):
    levels, seasonalities, log_diff_of_levels =  self.compute_levels_seasons(ts_objects)
    return levels, seasonalities, log_diff_of_levels
