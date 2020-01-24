import os
import numpy as np

import torch
import torch.nn as nn

from pathlib import Path
from src.DRNN import DRNN



class ES(nn.Module):
  def __init__(self, mc):
    super(ES, self).__init__()
    self.mc = mc
    self.max_num_series = self.mc.max_num_series
    self.seasonality = self.mc.seasonality
    self.output_size = self.mc.output_size

    # Level and Seasonality Smoothing parameters
    init_lev_sms = torch.ones((self.max_num_series, 1)) * 0.5
    init_seas_sms = torch.ones((self.max_num_series, 1)) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.seas_sms = nn.Parameter(data=init_seas_sms, requires_grad=True)

    init_seas = torch.ones((self.max_num_series, self.seasonality)) * 0.5
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

    # Lookup Smoothing parameterss
    init_lvl_sms = [self.lev_sms[idx] for idx in idxs]
    lev_sms = self.logistic(torch.stack(init_lvl_sms).squeeze(1))

    init_seas_sms = [self.seas_sms[idx] for idx in idxs]
    seas_sms = self.logistic(torch.stack(init_seas_sms).squeeze(1))

    init_seas_list = [torch.exp(self.init_seas[idx]) for idx in idxs]
    init_seas = torch.stack(init_seas_list)

    # Initialize seasonalities, levels and log_diff_of_levels
    seasonalities = []
    levels =[]

    for i in range(self.seasonality):
      seasonalities.append(init_seas[:,i])
    seasonalities.append(init_seas[:,0])
    levels.append(y[:,0]/seasonalities[0])

    for t in range(1, n_time):
      newlev = lev_sms * (y[:,t] / seasonalities[t]) + (1-lev_sms) * levels[t-1]
      newseason = seas_sms * (y[:,t] / newlev) + (1-seas_sms) * seasonalities[t]
      levels.append(newlev)
      seasonalities.append(newseason)

    seasonalities_stacked = torch.stack(seasonalities).transpose(1,0)
    levels_stacked = torch.stack(levels).transpose(1,0)

    # Completion of seasonalities if prediction horizon is larger than seasonality
    # Naive2 like prediction, to avoid recursive forecasting
    if self.output_size > self.seasonality:
      start_seasonality_ext = seasonalities_stacked.shape[1] - self.seasonality
      end_seasonality_ext = start_seasonality_ext + self.output_size - self.seasonality
      seasonalities = torch.cat((seasonalities_stacked,
                                 seasonalities_stacked[:, start_seasonality_ext:end_seasonality_ext]), 1)

    return levels_stacked, seasonalities_stacked


class RNN(nn.Module):
  def __init__(self, mc):
    super(RNN, self).__init__()
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
                   cell_type='LSTM')
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


class ESRNN(nn.Module):
  def __init__(self, mc):
    super(ESRNN, self).__init__()
    self.es = ES(mc)
    self.rnn = RNN(mc)
    self.mc = mc
  
  def gaussian_noise(self, input_data, is_training, std=0.2):
    if is_training:
      size = input_data.size()
      noise = torch.autograd.Variable(input_data.data.new(size).normal_(0, std))
      return input_data + noise
    else: return input_data
  
  def forward(self, ts_object, is_training=False):
    # parse mc
    batch_size = self.mc.batch_size
    input_size = self.mc.input_size
    output_size = self.mc.output_size
    exogenous_size = self.mc.exogenous_size
    noise_std = self.mc.noise_std

    # parse ts_object
    y_ts = ts_object.y
    idxs = ts_object.idxs
    n_series, n_time = y_ts.shape
    n_windows = n_time-input_size-output_size+1
    assert n_windows>0

    # Initialize windows, levels and seasonalities
    levels, seasonalities = self.es(ts_object)
    windows_x = torch.zeros((n_windows, batch_size, input_size+exogenous_size))
    windows_y = torch.zeros((n_windows, batch_size, output_size))
    for i in range(n_windows):
      x_start = i
      x_end = input_size+i

      # Deseasonalization and normalization
      x = y_ts[:, x_start:x_end] / seasonalities[:, x_start:x_end]
      x = x / levels[:, x_end]
      x = self.gaussian_noise(torch.log(x), is_training, std=noise_std)

      # Concatenate categories
      categories = ts_object.categories_vect
      x = torch.cat((x, categories), 1)

      y_start = x_end
      y_end = x_end+output_size

      # Deseasonalization and normalization
      y = y_ts[:, y_start:y_end] / seasonalities[:, y_start:y_end]
      y = torch.log(y) / levels[:, x_end]

      windows_x[i, :, :] += x
      windows_y[i, :, :] += y
    
    windows_y_hat = self.rnn(windows_x)
    return windows_y, windows_y_hat, levels
  
  def get_dir_name(self, root_dir=None):
    if not root_dir:
      assert self.mc.root_dir
      root_dir = self.mc.root_dir

    data_dir = self.mc.dataset_name
    model_parent_dir = os.path.join(root_dir, data_dir)
    model_path = ['num_series_{}'.format(self.mc.max_num_series),
                  'lr_{}'.format(self.mc.learning_rate),
                  str(self.mc.copy)]
    model_dir = os.path.join(model_parent_dir, '_'.join(model_path))
    return model_dir

  def save(self, model_dir=None, copy=None):
    if copy is not None:
      self.mc.copy = copy

    if not model_dir:
      assert self.mc.root_dir
      model_dir = self.get_dir_name()

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    
    rnn_filepath = os.path.join(model_dir, "rnn.model")
    es_filepath = os.path.join(model_dir, "es.model")

    print('Saving model to:\n {}'.format(model_dir)+'\n')
    torch.save({'model_state_dict': self.es.state_dict()}, es_filepath)
    torch.save({'model_state_dict': self.rnn.state_dict()}, rnn_filepath)

  def load(self, model_dir=None, copy=None):
    if copy is not None:
      self.mc.copy = copy

    if not model_dir:
      assert self.mc.root_dir
      model_dir = self.get_dir_name()
    
    rnn_filepath = os.path.join(model_dir, "rnn.model")
    es_filepath = os.path.join(model_dir, "es.model")
    path = Path(es_filepath)

    if path.is_file():
      print('Loading model from:\n {}'.format(model_dir)+'\n')

      checkpoint = torch.load(es_filepath, map_location=self.mc.device)
      self.es.load_state_dict(checkpoint['model_state_dict'])
      self.es.to(self.mc.device)
      
      checkpoint = torch.load(rnn_filepath, map_location=self.mc.device)
      self.rnn.load_state_dict(checkpoint['model_state_dict'])
      self.rnn.to(self.mc.device)
    else:
      print('Model path {} does not exist'.format(path))
