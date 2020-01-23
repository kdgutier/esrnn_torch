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
    self.mc = mc
    self.max_num_series = self.mc.max_num_series
    self.seasonality = self.mc.seasonality
    self.output_size = self.mc.output_size

  def compute_levels_seasons(self, ts_object):
  	pass

  def forward(self, x):
  	pass


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