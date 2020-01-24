import yaml
import os
from pathlib import Path

import numpy as np
import torch


class ModelConfig(object):
  def __init__(self, config_file, root_dir, copy=1):
      with open(config_file, 'r') as stream:
          config = yaml.safe_load(stream)
      
      # Train Parameters
      self.dataset_name = config['dataset_name']
      self.max_epochs = config['train_parameters']['max_epochs']
      self.freq_of_test = config['train_parameters']['freq_of_test']

      self.learning_rate = float(config['train_parameters']['learning_rate'])
      self.lr_scheduler_step_size = config['train_parameters']['lr_scheduler_step_size']
      self.per_series_lr_multip = config['train_parameters']['per_series_lr_multip']
      self.gradient_eps = float(config['train_parameters']['gradient_eps'])
      self.gradient_clipping_threshold = config['train_parameters']['gradient_clipping_threshold']
      self.noise_std = config['train_parameters']['noise_std']
      self.numeric_threshold = float(config['train_parameters']['numeric_threshold'])
      
      self.level_variability_penalty = config['train_parameters']['level_variability_penalty']
      self.c_state_penalty = config['train_parameters']['c_state_penalty']
      
      self.percentile = config['train_parameters']['percentile']
      self.training_percentile = config['train_parameters']['training_percentile']
      self.tau = self.percentile / 100.
      self.training_tau = self.training_percentile / 100.

      # Model Parameters
      self.state_hsize = config['model_parameters']['state_hsize']
      self.lback = config['model_parameters']['lback']
      self.dilations = config['model_parameters']['dilations']
      self.add_nl_layer = config['model_parameters']['add_nl_layer']
      self.attention_hsize = self.state_hsize
      self.averaging_level = config['model_parameters']['averaging_level']

      # Data Parameters
      self.seasonality = config['data_parameters']['seasonality']
      self.input_size = config['data_parameters']['input_size']
      self.input_size_i = self.input_size
      self.output_size = config['data_parameters']['output_size']
      self.output_size_i = self.output_size
      self.exogenous_size = config['data_parameters']['exogenous_size']

      self.min_inp_seq_length = config['data_parameters']['min_inp_seq_length']
      self.min_series_length = self.input_size_i + self.output_size_i + self.min_inp_seq_length + 2
      if self.seasonality == 4:
        self.max_series_length = (40 * self.seasonality) + self.min_series_length
      elif self.seasonality == 7:
        self.max_series_length = (20 * self.seasonality) + self.min_series_length
      elif self.seasonality == 12:
        self.max_series_length = (20 * self.seasonality) + self.min_series_length
      
      self.max_num_series = config['data_parameters']['max_num_series']
      self.data_dir = config['data_parameters']['data_dir']
      self.output_dir = config['data_parameters']['output_dir']

      self.root_dir = root_dir
      self.copy = copy
      self.device = "cpu"
  
  # def get_model(self):
  #   """
  #   Create Exponential Smoothing 
  #   Recursive Neural Network.
  #   """
  #   es = ES(mc=self)
  #   rnn = LSTM(mc=self)
  #   model = {'rnn': rnn, 'es': es}
  #   return model
