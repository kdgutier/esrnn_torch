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
    init_lev_sms = torch.ones((self.max_num_series, 1), dtype=torch.float64) * 0.5
    init_seas_sms = torch.ones((self.max_num_series, 1), dtype=torch.float64) * 0.5
    self.lev_sms = nn.Parameter(data=init_lev_sms, requires_grad=True)
    self.seas_sms = nn.Parameter(data=init_seas_sms, requires_grad=True)

    init_seas = torch.ones((self.max_num_series, self.seasonality), dtype=torch.float64) * 0.5
    self.init_seas = nn.Parameter(data=init_seas, requires_grad=True)
    self.logistic = nn.Sigmoid()

  def forward(self, ts_object):
    # Parse ts_object
    y = ts_object.y
    idxs = ts_object.idxs
    n_series, n_time = y.shape

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
      self.MLPB  = torch.nn.Parameter(torch.ones(mc.state_hsize))

    self.adapterW  = nn.Linear(mc.output_size, mc.state_hsize)
    self.adapterB  = torch.nn.Parameter(torch.ones(mc.output_size))
  
  def forward(self, y):
    for layer_num in range(len(self.rnn_stack)):
        residual = y
        y, _ = self.rnn_stack[layer_num](y)
        if layer_num > 0:
            out += residual
    
    if self.mc.add_nl_layer:
      y = torch.mul(self.MLPW, y) + self.MLPB
      y = torch.tanh(y)

    y = torch.mul(self.adapterW, rnn_ex) + self.adapterB
    return y


class ESRNN(object):
  def __init__(self, mc):
    self.es = ES(mc)
    self.rnn = RNN(mc)
    self.mc = mc

  def compute_levels_seasons(self, ts_objects):
    return self.es(ts_objects)
  
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
