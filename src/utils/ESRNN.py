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

    # Level and Seasonality Smoothing parameters
    init_lev_sms = torch.ones((self.n_series, 1)) * 0.5
    init_seas_sms = torch.ones((self.n_series, 1)) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.seas_sms = nn.Parameter(data=init_seas_sms, requires_grad=True)

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

    lev_sms = self.logistic(torch.stack(init_lvl_sms).squeeze(1))
    seas_sms = self.logistic(torch.stack(init_seas_sms).squeeze(1))

    init_seas_list = [torch.exp(self.init_seas[idx]) for idx in idxs]
    init_seas = torch.stack(init_seas_list)

    #print("init_seas.size()", init_seas.size())
    #print("lev_sms.size()", lev_sms.size())
    #print("seas_sms.size()", seas_sms.size())

    # Initialize seasonalities and levels
    seasonalities = []
    levels =[]
    for i in range(self.seasonality):
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

class _FastES(jit.ScriptModule):
  def __init__(self, mc):
    super(_FastES, self).__init__()
    self.mc = mc
    self.n_series = self.mc.n_series
    self.seasonality = self.mc.seasonality
    self.output_size = self.mc.output_size

    # Level and Seasonality Smoothing parameters
    init_sms = torch.ones((self.n_series, 2)) * 0.5
    self.sms = nn.Embedding(self.n_series, 2) #requires_grad=True default
    self.sms.weight.data.copy_(init_sms)

    init_seas = torch.ones((self.n_series, self.seasonality)) * 0.5
    self.init_seas = nn.Embedding(self.n_series, self.seasonality)
    self.sms.weight.data.copy_(init_seas)
    #self.logistic = nn.Sigmoid()

  @jit.script_method
  def forward(self, ts_object):
    """
    Computes levels and seasons
    """
    # Parse ts_object
    y = ts_object.y
    idxs = ts_object.idxs
    n_series, n_time = y.shape

    # Lookup Smoothings and Initial Seasonalities per serie
    sms = torch.sigmoid(self.sms(idxs))
    lev_sms, seas_sms = sms[:, 0], sms[:, 1]
    init_seas = torch.exp(self.init_seas(idxs))

    #print("init_seas.size()", init_seas.size())
    #print("lev_sms.size()", lev_sms.size())
    #print("seas_sms.size()", seas_sms.size())

    # Initialize seasonalities and levels
    seasonalities = torch.jit.annotate(List[Tensor], [])
    levels = torch.jit.annotate(List[Tensor], [])
    for i in range(self.seasonality):
      #seasonalities += 
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
    self.es = _ES(mc).to(self.mc.device)
    self.rnn = _RNN(mc).to(self.mc.device)

  def gaussian_noise(self, input_data, std=0.2):
    size = input_data.size()
    noise = torch.autograd.Variable(input_data.data.new(size).normal_(0, std))
    return input_data + noise

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
    levels, seasonalities = self.es(ts_object)
    windows_y_hat = torch.zeros((n_windows, batch_size, input_size+exogenous_size),
                                device=self.mc.device)
    windows_y = torch.zeros((n_windows, batch_size, output_size),
                            device=self.mc.device)

    for i, window in enumerate(windows_range):
      # Windows yhat
      y_hat_start = window
      y_hat_end = input_size + window
      # Deseasonalization and normalization
      window_y_hat = y[:, y_hat_start:y_hat_end] / seasonalities[:, y_hat_start:y_hat_end]
      window_y_hat = window_y_hat / levels[:, [y_hat_end-1]]
      window_y_hat = torch.log(window_y_hat)
      if self.training:
        window_y_hat = self.gaussian_noise(window_y_hat, std=noise_std)

      # Concatenate categories
      if exogenous_size>0:
        window_y_hat = torch.cat((window_y_hat, ts_object.categories), 1)

      # print('windows_y_hat.size',windows_y_hat.size())
      # print('window_y_hat.size',window_y_hat.size())
      # print('nwindows.size',n_windows)
    
      windows_y_hat[i, :, :] += window_y_hat

      # Windows y (for computing pinball loss during train)
      if self.training:
        y_start = y_hat_end
        y_end = y_start+output_size
        # Deseasonalization and normalization
        window_y = y[:, y_start:y_end] / seasonalities[:, y_start:y_end]
        window_y = window_y / levels[:, [y_start]]
        window_y = torch.log(window_y)
        windows_y[i, :, :] += window_y

    # RNN Forward
    windows_y_hat = self.rnn(windows_y_hat)
    
    if self.training:
      return windows_y, windows_y_hat, levels
    else:
      y_hat = windows_y_hat[-1,:,:]
      trends = torch.exp(y_hat)

      # Completion of seasonalities if prediction horizon is larger than seasonality
      # Naive2 like prediction, to avoid recursive forecasting
      if output_size > seasonality:
        repetitions = int(np.ceil(output_size/seasonality))-1
        last_season = seasonalities[:, -seasonality:]
        extra_seasonality = last_season.repeat((1, repetitions))
        seasonalities = torch.cat((seasonalities, extra_seasonality), 1)
      # Deseasonalization and normalization (inverse)
      y_hat = trends * levels[:, [n_time-1]] * seasonalities[:, n_time:(n_time+output_size)]
      return y_hat