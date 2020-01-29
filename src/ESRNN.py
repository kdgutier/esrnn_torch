import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pathlib import Path
from src.utils.config import ModelConfig
from src.utils.ESRNN import _ESRNN
from src.utils.losses import SmylLoss
from src.utils.data import Iterator


class ESRNN(object):
    def __init__(self, max_epochs=15, batch_size=1, learning_rate=1e-3, per_series_lr_multip=1, gradient_eps=1e-6, gradient_clipping_threshold=20,
                 noise_std=0.001, level_variability_penalty=80, tau=0.5, state_hsize=40, dilations=[[1, 2], [4, 8]], add_nl_layer=False, 
                 seasonality=4, input_size=4, output_size=8, frequency='D', max_periods=20, root_dir='./'):
        super(ESRNN, self).__init__()
        self.mc = ModelConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=learning_rate, per_series_lr_multip=per_series_lr_multip, 
                              gradient_eps=gradient_eps, gradient_clipping_threshold=gradient_clipping_threshold, 
                              noise_std=noise_std, level_variability_penalty=level_variability_penalty, tau=tau,
                              state_hsize=state_hsize, dilations=dilations, add_nl_layer=add_nl_layer, 
                              seasonality=seasonality, input_size=input_size, output_size=output_size,
                              frequency=frequency, max_periods=max_periods, root_dir=root_dir)

    def train(self, dataloader, random_seed):
      print(10*'='+' Training ESRNN ' + 10*'=' + '\n')

      # Optimizers
      # TODO scheduler
      es_optimizer = optim.Adam(params=self.esrnn.es.parameters(),
                                  lr=self.mc.learning_rate*self.mc.per_series_lr_multip, 
                                  betas=(0.9, 0.999), eps=self.mc.gradient_eps)

      rnn_optimizer = optim.Adam(params=self.esrnn.rnn.parameters(),
                                  lr=self.mc.learning_rate,
                                  betas=(0.9, 0.999), eps=self.mc.gradient_eps)
      
      # Loss Functions
      smyl_loss = SmylLoss(tau=self.mc.tau, level_variability_penalty=self.mc.level_variability_penalty)

      # training code
      for epoch in range(self.mc.max_epochs):
        start = time.time()
        
        losses = []
        for j in range(dataloader.n_batches):
          es_optimizer.zero_grad()
          rnn_optimizer.zero_grad()

          batch = dataloader.get_batch()
          windows_y, windows_y_hat, levels = self.esrnn(batch)
          
          loss = smyl_loss(windows_y, windows_y_hat, levels)
          losses.append(loss.data.numpy())
          loss.backward()
          torch.nn.utils.clip_grad_value_(self.esrnn.rnn.parameters(),
                                      clip_value=self.mc.gradient_clipping_threshold)
          torch.nn.utils.clip_grad_value_(self.esrnn.es.parameters(),
                                      clip_value=self.mc.gradient_clipping_threshold)
          rnn_optimizer.step()
          es_optimizer.step()

        print("========= Epoch {} finished =========".format(epoch))
        print("Training time: {}".format(time.time()-start))
        print("Forecast loss: {}".format(np.mean(losses)))

      print('Train finished!')
    
    def fit(self, X_df, y_df, random_seed=1):
      # Transform long dfs to wide numpy
      assert type(X_df) == pd.core.frame.DataFrame
      assert type(y_df) == pd.core.frame.DataFrame
      assert all([(col in X_df) for col in ['unique_id', 'ds', 'x']])
      assert all([(col in y_df) for col in ['unique_id', 'ds', 'y']])

      X, y = self.long_to_wide(X_df, y_df)
      assert len(X)==len(y)
      assert X.shape[1]>=3
      
      # Random Seeds
      torch.manual_seed(random_seed)
      np.random.seed(random_seed)

      # Exogenous variables
      unique_categories = np.unique(X[:, 1])
      self.mc.category_to_idx = dict((word, index) for index, word in enumerate(unique_categories))
      self.mc.exogenous_size = len(unique_categories)

      # Create batches
      self.dataloader = Iterator(mc=self.mc, X=X, y=y, shuffle=False)
      self.sort_key = self.dataloader.sort_key

      # Initialize model
      self.mc.n_series = self.dataloader.n_series

      self.esrnn = _ESRNN(self.mc)

      # Train model
      self.train(dataloader=self.dataloader, random_seed=random_seed)

    def predict(self, X_df):
      """
          Predictions for all stored time series
      Returns:
          Y_hat_panel : array-like (n_samples, 1).
              Predicted values for models in Family for ids in Panel.
          ds: Corresponding list of date stamps
          unique_id: Corresponding list of unique_id
      """
      # Obtain unique_ids to predict
      predict_unique_idxs = X_df['unique_id'].unique()

      # Predictions for panel
      Y_hat_panel = pd.DataFrame(columns=['unique_id', 'y_hat'])

      for unique_id in predict_unique_idxs:
        # Corresponding train batch
        batch = self.dataloader.get_batch(unique_id=unique_id)

        # Declare y_hat_id placeholder
        Y_hat_id = pd.DataFrame(np.zeros(shape=(self.mc.output_size, 1)), columns=["y_hat"])

        # Prediction
        y_hat = self.esrnn.predict(batch)
        y_hat = y_hat.squeeze()
        Y_hat_id.iloc[:, 0] = y_hat

        # Serie prediction
        Y_hat_id["unique_id"] = unique_id
        ts = date_range = pd.date_range(start=batch.last_ds[0],
                                        periods=self.mc.output_size+1, freq=self.mc.frequency)
        Y_hat_id["ts"] = ts[1:]
        Y_hat_panel = Y_hat_panel.append(Y_hat_id, sort=False).reset_index(drop=True)

      return Y_hat_panel
    
    def long_to_wide(self, X_df, y_df):
      data = X_df
      data['y'] = y_df['y']
      ds_map = {}
      for dmap, t in enumerate(data['ds'].unique()):
          ds_map[t] = dmap
      data['ds_map'] = data['ds'].map(ds_map)
      df_wide = data.pivot(index='unique_id', columns='ds_map')['y']
      
      x_unique = data[['unique_id', 'x']].groupby('unique_id').first()
      last_ds =  data[['unique_id', 'ds']].groupby('unique_id').last()
      assert len(x_unique)==len(data.unique_id.unique())
      df_wide['x'] = x_unique
      df_wide['last_ds'] = last_ds
      df_wide = df_wide.reset_index().rename_axis(None, axis=1)
      
      ds_cols = data.ds_map.unique().tolist()
      X = df_wide.filter(items=['unique_id', 'x', 'last_ds']).values
      y = df_wide.filter(items=ds_cols).values

      # TODO: assert "completeness" of the series (frequency-wise)
      return X, y
  
    def get_dir_name(self, root_dir=None):
      if not root_dir:
        assert self.mc.root_dir
        root_dir = self.mc.root_dir

      data_dir = self.mc.dataset_name
      model_parent_dir = os.path.join(root_dir, data_dir)
      model_path = ['num_series_{}'.format(self.mc.num_series),
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
