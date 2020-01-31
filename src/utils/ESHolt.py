import torch
import torch.nn as nn
import numpy as np


class _ESHolt(nn.Module):
  def __init__(self, mc):
    super(_ES, self).__init__()
    self.mc = mc
    self.n_series = self.mc.n_series
    self.seasonality = self.mc.seasonality
    self.output_size = self.mc.output_size

    # Level and Seasonality Smoothing parameters
    init_lev_sms = torch.ones((self.n_series, 1)) * 0.5
    init_seas_sms = torch.ones((self.n_series, 1)) * 0.5
    init_trend_sms = torch.ones((self.n_series, 1)) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.seas_sms = nn.Parameter(data=init_seas_sms, requires_grad=True)
    self.trend_sms = nn.Parameter(data=init_trend_sms, requires_grad=True)

    init_seas = torch.ones((self.n_series, self.seasonality)) * 0.5
    self.init_seas = nn.Parameter(data=init_seas, requires_grad=True)
    self.logistic = nn.Sigmoid()

  def forward(self, ts_object):
    """
    Computes levels and seasons
    """
    # Parse ts_object
    y = ts_object.y
    idxs = ts_object.idxs
    n_series, n_time = y.shape

    # Lookup Smoothing parameters per serie
    init_lvl_sms = [self.lev_sms[idx] for idx in idxs]
    init_seas_sms = [self.seas_sms[idx] for idx in idxs]
    init_trend_sms = [self.trend_sms[idx] for idx in idxs]

    lev_sms = self.logistic(torch.stack(init_lvl_sms).squeeze(1))
    seas_sms = self.logistic(torch.stack(init_seas_sms).squeeze(1))
    trend_sms = self.logistic(torch.stack(init_seas_sms).squeeze(1))

    init_seas_list = [torch.exp(self.init_seas[idx]) for idx in idxs]
    init_seas = torch.stack(init_seas_list)

    # Initialize seasonalities and levels
    seasonalities = []
    levels = []
    trends = []

    for i in range(self.seasonality):
      seasonalities.append(init_seas[:,i])
    seasonalities.append(init_seas[:,0])
    levels.append(y[:,0]/seasonalities[0])
    trends.append((y[:,1]/seasonalities[1] - levels[0])/2)

    # Recursive seasonalities and levels
    for t in range(1, n_time):
      newlev = lev_sms*(y[:,t] / seasonalities[t]) + (1-lev_sms)*(levels[t-1] + trend[t-1])
      newtrend = trend_sms*(levels[t]-levels[t-1]) + (1-trend_sms)*trend[t-1]
      newseason = seas_sms*(y[:,t] / (newlev+new_trend)) + (1-seas_sms) * seasonalities[t]
      levels.append(newlev)
      trends.append(newtrend)
      seasonalities.append(newseason)
    
    levels_stacked = torch.stack(levels).transpose(1,0)
    seasonalities_stacked = torch.stack(seasonalities).transpose(1,0)
    trends_stacked = torch.stack(trends).transpose(1,0)

    return levels_stacked, seasonalities_stacked, trends
