import torch
import torch.nn as nn
from src.utils.DRNN import DRNN
import numpy as np

import torch.jit as jit


class _ES(nn.Module):
  def __init__(self, mc):
    super(_ES, self).__init__()
    self.mc = mc
    self.n_series = self.mc.n_series
    self.seasonality = self.mc.seasonality
    self.output_size = self.mc.output_size
  
  def gaussian_noise(self, input_data, std=0.2):
    size = input_data.size()
    noise = torch.autograd.Variable(input_data.data.new(size).normal_(0, std))
    return input_data + noise
  
  def compute_levels_seasons(self, ts_object):
    pass

  def normalize(self, y, level, seasonalities):
    pass
  
  def predict(self, trend, levels, seasonalities):
    pass
  
  def forward(self, ts_object):
    # parse mc
    input_size = self.mc.input_size
    output_size = self.mc.output_size
    exogenous_size = self.mc.exogenous_size
    noise_std = self.mc.noise_std
    seasonality = self.mc.seasonality
    batch_size = len(ts_object.idxs)

    # parse ts_object
    y = ts_object.y
    n_time = y.shape[1]
    if self.training:
      windows_end = n_time-input_size-output_size+1
      windows_range = range(windows_end)
    else:
      windows_start = n_time-input_size-output_size+1
      windows_end = n_time-input_size+1
      
      windows_range = range(windows_start, windows_end)
    n_windows = len(windows_range)
    assert n_windows>0

    # Initialize windows, levels and seasonalities
    levels, seasonalities = self.compute_levels_seasons(ts_object)
    windows_y_hat = torch.zeros((n_windows, batch_size, input_size+exogenous_size),
                                device=self.mc.device)
    windows_y = torch.zeros((n_windows, batch_size, output_size),
                            device=self.mc.device)

    for i, window in enumerate(windows_range):
      # Windows yhat
      y_hat_start = window
      y_hat_end = input_size + window
      
      # Y_hat deseasonalization and normalization
      window_y_hat = self.normalize(y=y[:, y_hat_start:y_hat_end],
                                    level=levels[:, [y_hat_end-1]],
                                    seasonalities=seasonalities,
                                    start=y_hat_start, end=y_hat_end)

      if self.training:
        window_y_hat = self.gaussian_noise(window_y_hat, std=noise_std)

      # Concatenate categories
      if exogenous_size>0:
        window_y_hat = torch.cat((window_y_hat, ts_object.categories), 1)
    
      windows_y_hat[i, :, :] += window_y_hat

      # Windows y (for loss during train)
      if self.training:
        y_start = y_hat_end
        y_end = y_start+output_size
        # Y deseasonalization and normalization
        window_y = self.normalize(y=y[:, y_start:y_end],
                                  level=levels[:, [y_start]],
                                  seasonalities=seasonalities,
                                  start=y_start, end=y_end)
        windows_y[i, :, :] += window_y

    return windows_y_hat, windows_y, levels, seasonalities

class _ES0(_ES):
  def __init__(self, mc):
    super(_ES0, self).__init__(mc)
    # Level Smoothing parameters
    init_lev_sms = torch.ones((self.n_series, 1)) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.logistic = nn.Sigmoid()
  
  def compute_levels_seasons(self, ts_object):
    """
    Computes levels and seasons
    """
    # Parse ts_object
    y = ts_object.y
    idxs = ts_object.idxs
    n_series, n_time = y.shape

    # Lookup Smoothing parameters per serie
    init_lvl_sms = [self.lev_sms[idx] for idx in idxs]
    lev_sms = self.logistic(torch.stack(init_lvl_sms).squeeze(1))

    # Initialize levels
    levels =[]
    levels.append(y[:,0])

    # Recursive seasonalities and levels
    for t in range(1, n_time):
      newlev = lev_sms * (y[:,t]) + (1-lev_sms) * levels[t-1]
      levels.append(newlev)
    
    levels = torch.stack(levels).transpose(1,0)
    seasonalities = None

    return levels, seasonalities
  
  def normalize(self, y, level, seasonalities, start, end):
    # Normalization
    y = y / level
    y = torch.log(y)
    return y
  
  def predict(self, trend, levels, seasonalities):
    n_time = levels.shape[1]

    # Denormalization
    trend = torch.exp(trend)
    y_hat = trend * levels[:,[n_time-1]]
    return y_hat


class _ES1(_ES):
  def __init__(self, mc):
    super(_ES1, self).__init__(mc)
    # Level and Seasonality Smoothing parameters
    init_lev_sms = torch.ones((self.n_series, 1)) * 0.5
    init_seas_sms = torch.ones((self.n_series, 1)) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.seas_sms = nn.Parameter(data=init_seas_sms, requires_grad=True)

    init_seas = torch.ones((self.n_series, self.seasonality[0])) * 0.5
    self.init_seas = nn.Parameter(data=init_seas, requires_grad=True)
    self.logistic = nn.Sigmoid()

  def compute_levels_seasons(self, ts_object):
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

    lev_sms = self.logistic(torch.stack(init_lvl_sms).squeeze(1))
    seas_sms = self.logistic(torch.stack(init_seas_sms).squeeze(1))

    # Initialize seasonalities and levels
    init_seas_list = [torch.exp(self.init_seas[idx]) for idx in idxs]
    init_seas = torch.stack(init_seas_list)

    seasonalities = []
    levels =[]
    for i in range(self.seasonality[0]):
      seasonalities.append(init_seas[:,i])
    seasonalities.append(init_seas[:,0])
    levels.append(y[:,0]/seasonalities[0])

    # Recursive seasonalities and levels
    for t in range(1, n_time):
      newlev = lev_sms * (y[:,t] / seasonalities[t]) + (1-lev_sms) * levels[t-1]
      newseason = seas_sms * (y[:,t] / newlev) + (1-seas_sms) * seasonalities[t]
      levels.append(newlev)
      seasonalities.append(newseason)
    
    levels = torch.stack(levels).transpose(1,0)
    seasonalities = torch.stack(seasonalities).transpose(1,0)

    return levels, seasonalities

  def normalize(self, y, level, seasonalities, start, end):
    # Deseasonalization and normalization
    y_n = y / seasonalities[:, start:end]
    y_n = y_n / level
    y_n = torch.log(y_n)
    return y_n
  
  def predict(self, trend, levels, seasonalities):
    output_size = self.mc.output_size
    seasonality = self.mc.seasonality[0]
    n_time = levels.shape[1]

    # Denormalize
    trend = torch.exp(trend)

    # Completion of seasonalities if prediction horizon is larger than seasonality
    # Naive2 like prediction, to avoid recursive forecasting
    if output_size > seasonality:
      repetitions = int(np.ceil(output_size/seasonality))-1
      last_season = seasonalities[:, -seasonality:]
      extra_seasonality = last_season.repeat((1, repetitions))
      seasonalities = torch.cat((seasonalities, extra_seasonality), 1)
    # Deseasonalization and normalization (inverse)
    y_hat = trend * levels[:,[n_time-1]] * seasonalities[:, n_time:(n_time+output_size)]

    return y_hat


class _ES2(_ES):
  def __init__(self, mc):
    super(_ES2, self).__init__(mc)
    # Level and Seasonality Smoothing parameters
    init_lev_sms = torch.ones((self.n_series, 1)) * 0.5
    init_seas_sms1 = torch.ones((self.n_series, 1)) * 0.5
    init_seas_sms2 = torch.ones((self.n_series, 1)) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.seas_sms1 = nn.Parameter(data=init_seas_sms1, requires_grad=True)
    self.seas_sms2 = nn.Parameter(data=init_seas_sms2, requires_grad=True)

    init_seas1 = torch.ones((self.n_series, self.seasonality[0])) * 0.5
    self.init_seas1 = nn.Parameter(data=init_seas1, requires_grad=True)
    init_seas2 = torch.ones((self.n_series, self.seasonality[1])) * 0.5
    self.init_seas2 = nn.Parameter(data=init_seas2, requires_grad=True)
    self.logistic = nn.Sigmoid()
  
  def compute_levels_seasons(self, ts_object):
    # Parse ts_object
    y = ts_object.y
    idxs = ts_object.idxs
    n_series, n_time = y.shape

    # Lookup Smoothing parameters per serie
    init_lvl_sms = [self.lev_sms[idx] for idx in idxs]
    init_seas_sms1 = [self.seas_sms1[idx] for idx in idxs]
    init_seas_sms2 = [self.seas_sms2[idx] for idx in idxs]

    lev_sms = self.logistic(torch.stack(init_lvl_sms).squeeze(1))
    seas_sms1 = self.logistic(torch.stack(init_seas_sms1).squeeze(1))
    seas_sms2 = self.logistic(torch.stack(init_seas_sms2).squeeze(1))

    # Initialize seasonalities1
    init_seas_list1 = [torch.exp(self.init_seas1[idx]) for idx in idxs]
    init_seas1 = torch.stack(init_seas_list1)

    seasonalities1 = []
    for i in range(self.seasonality[0]):
      seasonalities1.append(init_seas1[:,i])
    seasonalities1.append(init_seas1[:,0])

    # Initialize seasonalities2
    init_seas_list2 = [torch.exp(self.init_seas2[idx]) for idx in idxs]
    init_seas2 = torch.stack(init_seas_list2)

    seasonalities2 = []
    for i in range(self.seasonality[1]):
      seasonalities2.append(init_seas2[:,i])
    seasonalities2.append(init_seas2[:,0])

    levels = []
    levels.append(y[:,0]/(seasonalities1[0] * seasonalities2[0]))

    # Recursive seasonalities and levels
    for t in range(1, n_time):
      newlev = lev_sms * (y[:,t] / (seasonalities1[t] * seasonalities2[t])) + \
               (1-lev_sms) * levels[t-1]
      newseason1 = seas_sms1 * (y[:,t] / (newlev * seasonalities2[t])) + \
               (1-seas_sms1) * seasonalities1[t]
      newseason2 = seas_sms2 * (y[:,t] / (newlev * seasonalities1[t])) + \
               (1-seas_sms2) * seasonalities2[t]
      levels.append(newlev)
      seasonalities1.append(newseason1)
      seasonalities2.append(newseason2)
    
    levels = torch.stack(levels).transpose(1,0)
    seasonalities1 = torch.stack(seasonalities1).transpose(1,0)
    seasonalities2 = torch.stack(seasonalities2).transpose(1,0)

    seasonalities = (seasonalities1, seasonalities2)
    return levels, seasonalities
  
  def normalize(self, y, level, seasonalities, start, end):
    # Deseasonalization and normalization
    y_n = y / seasonalities[0][:, start:end]
    y_n = y_n / seasonalities[1][:, start:end]
    y_n = y_n / level
    y_n = torch.log(y_n)
    return y_n
  
  def predict(self, trend, levels, seasonalities):
    output_size = self.mc.output_size
    seasonality1 = self.mc.seasonality[0]
    seasonality2 = self.mc.seasonality[1]
    n_time = levels.shape[1]

    # Denormalize
    trend = torch.exp(trend)

    # Completion of seasonalities if prediction horizon is larger than seasonality
    # Naive2 like prediction, to avoid recursive forecasting
    if output_size > seasonality1:
      repetitions = int(np.ceil(output_size/seasonality1))-1
      last_season = seasonalities[0][:, -seasonality1:]
      extra_seasonality = last_season.repeat((1, repetitions))
      seasonalities[0] = torch.cat((seasonalities[0], extra_seasonality), 1)

    if output_size > seasonality2:
      repetitions = int(np.ceil(output_size/seasonality2))-1
      last_season = seasonalities[1][:, -seasonality2:]
      extra_seasonality = last_season.repeat((1, repetitions))
      seasonalities[1] = torch.cat((seasonalities[1], extra_seasonality), 1)

    # Deseasonalization and normalization (inverse)
    y_hat = trend * levels[:,[n_time-1]] * \
            seasonalities[0][:, n_time:(n_time+output_size)] * \
            seasonalities[1][:, n_time:(n_time+output_size)]

    return y_hat


class _RNN(nn.Module):
  def __init__(self, mc):
    super(_RNN, self).__init__()
    self.mc = mc
    self.layers = len(mc.dilations)

    layers = []
    for grp_num in range(len(mc.dilations)):
      if grp_num == 0:
        input_size = mc.input_size + mc.exogenous_size
      else:
        input_size = mc.state_hsize
      layer = DRNN(input_size,
                   mc.state_hsize,
                   n_layers=1,
                   dilations=mc.dilations[grp_num],
                   cell_type=mc.cell_type)
      layers.append(layer)

    self.rnn_stack = nn.Sequential(*layers)

    if self.mc.add_nl_layer:
      self.MLPW  = nn.Linear(mc.state_hsize, mc.state_hsize)

    self.adapterW  = nn.Linear(mc.state_hsize, mc.output_size)
  
  def forward(self, input_data):
    for layer_num in range(len(self.rnn_stack)):
      residual = input_data
      output, _ = self.rnn_stack[layer_num](input_data)
      if layer_num > 0:
        output += residual
      input_data = output

    if self.mc.add_nl_layer:
      input_data = self.MLPW(input_data)
      input_data = torch.tanh(input_data)

    input_data = self.adapterW(input_data)
    return input_data


class _ESRNN(nn.Module):
  def __init__(self, mc):
    super(_ESRNN, self).__init__()
    self.mc = mc
    if len(mc.seasonality)==0:
      self.es = _ES0(mc).to(self.mc.device)
    elif len(mc.seasonality)==1:
      self.es = _ES1(mc).to(self.mc.device)
    elif len(mc.seasonality)==2:
      self.es = _ES2(mc).to(self.mc.device)
    
    self.rnn = _RNN(mc).to(self.mc.device)

  def forward(self, ts_object):
    # ES Forward
    windows_y_hat, windows_y, levels, seasonalities = self.es(ts_object)

    # RNN Forward
    windows_y_hat = self.rnn(windows_y_hat)
    
    return windows_y, windows_y_hat, levels
  
  def predict(self, ts_object):
    # ES Forward
    windows_y_hat, _, levels, seasonalities = self.es(ts_object)

    # RNN Forward
    windows_y_hat = self.rnn(windows_y_hat)
    trend = windows_y_hat[-1,:,:] # Last observation prediction
    
    y_hat = self.es.predict(trend, levels, seasonalities)    
    return y_hat
