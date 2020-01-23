import os
import numpy as np

import torch
import torch.nn as nn

class VanillaLSTM(nn.Module):
  def __init__(self, mc):
      self.mc = mc
      input_size = mc.input_size
      state_hsize= mc.state_hsize 
      output_size = mc.output_size
      dilations = mc.dilations
      exogenous_size = mc.exogenous_size
      self.layers = len(mc.dilations)

  def forward(self, x):
  	pass


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

  def forward(self, ts_objects):
    # Parse ts_objects
    idxs = ts_objects.idxs
    y = ts_objects.y
    n_series, n_time = y.shape
    assert n_series == len(idxs)

    # Lookup embeddings
    lev_sms = nn.Sigmoid(self.lev_sms[idxs])
    seas_sms = nn.Sigmoid(self.lev_sms[idxs])
    init_seas = torch.exp(self.init_seas[idxs, :])

    # Initialize seasonalities, levels and log_diff_of_levels
    seasonalities = torch.zeros((n_series, self.seasonality+n_time))
    levels = torch.zeros((n_series, n_time))
    log_diff_of_levels = torch.zeros((n_series, n_time-1))

    seasonalities[:, :self.seasonality+1] = torch.cat((init_seas, init_seas[:,[0]]), 1)
    levels[:, 0] = y / seasonalities[:, 0]

    for t in range(1, n_time):
      newlev = lev_sms * (y[:, t] / seasonalities[:, t]) + (1-lev_sms) * levels[:, t-1]
      newseason = seas_sms * (y[:, t] / newlev_ex) + (1-seas_sms_ex) * seasonalities[:, t]
      log_diff = torch.log(newlev)-torch.log(levels[:, t-1])
      
      seasonalities[:, t+self.seasonality+1] += newseason
      levels[:, t] += newlev
      log_diff_of_levels[:, t-1] += log_diff

    # Completion of seasonalities if prediction horizon is larger than seasonality
    # Naive2 like prediction, to avoid recursive forecasting 
    if self.output_size > self.seasonality:
      start_seasonality_ext = len(seasonalities) - self.seasonality
      end_seasonality_ext = start_seasonality_ext + self.output_size - self.seasonality
      seasonalities = torch.cat((seasonalities,
                                 seasonalities[:,[start_seasonality_ext:end_seasonality_ext]]), 1)

  	return levels, seasonalities, log_diff_of_levels

class ESRNN(object):
  def __init__(self, mc):
    self.es = ES(mc)
    self.rnn = VanillaLSTM(mc)
    self.mc = mc

  def compute_levels_seasons(self, ts_object):
    return self.es.compute_levels_seasons(ts_object)

  def forward(self, x):
  	pass
  
  def predict(self, ts_object):
    pass

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
    self.rnn.pc.save(rnn_filepath)
    self.es.pc.save(es_filepath)

  def load(self, model_dir=None, copy=None):
    if copy is not None:
      self.mc.copy = copy

    if not model_dir:
      assert self.mc.root_dir
      model_dir = self.get_dir_name()
    
    rnn_filepath = os.path.join(model_dir, "rnn.model")
    es_filepath = os.path.join(model_dir, "es.model")
    path = Path(rnn_filepath)

    if path.is_file():
      print('Loading model from:\n {}'.format(model_dir)+'\n')
      self.rnn.pc.populate(rnn_filepath)
      self.es.pc.populate(es_filepath)
    else:
      print('Model path {} does not exist'.format(path))