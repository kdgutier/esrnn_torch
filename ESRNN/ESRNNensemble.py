import os
import time
import random

import numpy as np
import pandas as pd

import torch

from pathlib import Path

from ESRNN.utils.config import ModelConfig
from ESRNN.utils.losses import DisaggregatedPinballLoss
from ESRNN.utils.data import Iterator

from ESRNN.ESRNN import ESRNN

from ESRNN.utils_evaluation import owa

class ESRNNensemble(object):
  """ Exponential Smoothing Recursive Neural Network Ensemble.
    n_models=1
    n_top=1
  """
  def __init__(self, n_models=1, n_top=1, max_epochs=15, batch_size=1, batch_size_test=128,
               freq_of_test=-1, learning_rate=1e-3, lr_scheduler_step_size=9, lr_decay=0.9,
               per_series_lr_multip=1.0, gradient_eps=1e-8, gradient_clipping_threshold=20,
               rnn_weight_decay=0, noise_std=0.001, level_variability_penalty=80,
               testing_percentile=50, training_percentile=50, ensemble=False, cell_type='LSTM',
               state_hsize=40, dilations=[[1, 2], [4, 8]],
               add_nl_layer=False, seasonality=[4], input_size=4, output_size=8,
               frequency='D', max_periods=20, random_seed=1,
               device='cuda', root_dir='./'):
    super(ESRNNensemble, self).__init__()

    self.n_models = n_models
    self.n_top = n_top
    assert n_models>=2, "Number of models for ensemble should be greater than 1"
    assert n_top<=n_models, "Number of top models should be smaller than models to ensemble"
    self.big_float = 1e6
    self.mc = ModelConfig(max_epochs=max_epochs, batch_size=batch_size, batch_size_test=batch_size_test,
                          freq_of_test=freq_of_test, learning_rate=learning_rate,
                          lr_scheduler_step_size=lr_scheduler_step_size, lr_decay=lr_decay,
                          per_series_lr_multip=per_series_lr_multip,
                          gradient_eps=gradient_eps, gradient_clipping_threshold=gradient_clipping_threshold,
                          rnn_weight_decay=rnn_weight_decay, noise_std=noise_std,
                          level_variability_penalty=level_variability_penalty,
                          testing_percentile=testing_percentile, training_percentile=training_percentile,
                          ensemble=ensemble, cell_type=cell_type,
                          state_hsize=state_hsize, dilations=dilations, add_nl_layer=add_nl_layer,
                          seasonality=seasonality, input_size=input_size, output_size=output_size,
                          frequency=frequency, max_periods=max_periods, random_seed=random_seed,
                          device=device, root_dir=root_dir)
    self._fitted = False

  def fit(self, X_df, y_df, X_test_df=None, y_test_df=None, shuffle=True):
    # Transform long dfs to wide numpy
    assert type(X_df) == pd.core.frame.DataFrame
    assert type(y_df) == pd.core.frame.DataFrame
    assert all([(col in X_df) for col in ['unique_id', 'ds', 'x']])
    assert all([(col in y_df) for col in ['unique_id', 'ds', 'y']])

    # Storing dfs for OWA evaluation, initializing min_owa
    self.y_train_df = y_df
    self.X_test_df = X_test_df
    self.y_test_df = y_test_df
    self.min_owa = 4.0
    self.min_epoch = 0

    # Exogenous variables
    unique_categories = X_df['x'].unique()
    self.mc.category_to_idx = dict((word, index) for index, word in enumerate(unique_categories))
    self.mc.exogenous_size = len(unique_categories)

    self.unique_ids = X_df['unique_id'].unique()
    self.mc.n_series = len(self.unique_ids)

    # Set seeds
    torch.manual_seed(self.mc.random_seed)
    np.random.seed(self.mc.random_seed)

    # Initial series random assignment to models
    self.series_models_map = np.zeros((self.mc.n_series, self.n_models))
    n_initial_models = int(np.ceil(self.n_models/2))
    for i in range(self.mc.n_series):
      id_models = np.random.choice(self.n_models, n_initial_models)
      self.series_models_map[i,id_models] = 1

    self.esrnn_ensemble = []
    for _ in range(self.n_models):
      esrnn = ESRNN(max_epochs=self.mc.max_epochs, batch_size=self.mc.batch_size, batch_size_test=self.mc.batch_size_test,
                    freq_of_test=-1, learning_rate=self.mc.learning_rate,
                    lr_scheduler_step_size=self.mc.lr_scheduler_step_size, lr_decay=self.mc.lr_decay,
                    per_series_lr_multip=self.mc.per_series_lr_multip,
                    gradient_eps=self.mc.gradient_eps, gradient_clipping_threshold=self.mc.gradient_clipping_threshold,
                    rnn_weight_decay=self.mc.rnn_weight_decay, noise_std=self.mc.noise_std,
                    level_variability_penalty=self.mc.level_variability_penalty,
                    testing_percentile=self.mc.testing_percentile,
                    training_percentile=self.mc.training_percentile, ensemble=self.mc.ensemble,
                    cell_type=self.mc.cell_type,
                    state_hsize=self.mc.state_hsize, dilations=self.mc.dilations, add_nl_layer=self.mc.add_nl_layer,
                    seasonality=self.mc.seasonality, input_size=self.mc.input_size, output_size=self.mc.output_size,
                    frequency=self.mc.frequency, max_periods=self.mc.max_periods, random_seed=self.mc.random_seed,
                    device=self.mc.device, root_dir=self.mc.root_dir)

      # To instantiate _ESRNN object within ESRNN class we need n_series
      esrnn.instantiate_esrnn(self.mc.exogenous_size, self.mc.n_series)
      esrnn._fitted = True
      self.esrnn_ensemble.append(esrnn)

    self.X, self.y = esrnn.long_to_wide(X_df, y_df)
    assert len(self.X)==len(self.y)
    assert self.X.shape[1]>=3

    # Train model
    self._fitted = True
    self.train()

  def train(self):
    # Initial performance matrix
    self.performance_matrix = np.ones((self.mc.n_series, self.n_models)) * self.big_float
    warm_start = False
    train_tau = self.mc.training_percentile/100
    criterion = DisaggregatedPinballLoss(train_tau)

    # Train epoch loop
    for epoch in range(self.mc.max_epochs):
      start = time.time()

      # Solve degenerate models
      for model_id in range(self.n_models):
        if np.sum(self.series_models_map[:,model_id])==0:
          print('Reassigning random series to model ', model_id)
          n_sample_series= int(self.mc.n_series/2)
          index_series = np.random.choice(self.mc.n_series, n_sample_series, replace=False)
          self.series_models_map[index_series, model_id] = 1

      # Model loop
      for model_id, esrnn in enumerate(self.esrnn_ensemble):
        # Train model with subset data
        dataloader = Iterator(mc = self.mc, X=self.X, y=self.y,
                                      weights=self.series_models_map[:, model_id])
        esrnn.train(dataloader, max_epochs=1, warm_start=warm_start, shuffle=True, verbose=False)

        # Compute model performance for each series
        dataloader = Iterator(mc=self.mc, X=self.X, y=self.y)
        per_series_evaluation = esrnn.per_series_evaluation(dataloader, criterion=criterion)
        self.performance_matrix[:, model_id] = per_series_evaluation

      # Reassign series to models
      self.series_models_map = np.zeros((self.mc.n_series, self.n_models))
      top_models = np.argpartition(self.performance_matrix, self.n_top)[:, :self.n_top]
      for i in range(self.mc.n_series):
        self.series_models_map[i, top_models[i,:]] = 1

      warm_start = True

      print("========= Epoch {} finished =========".format(epoch))
      print("Training time: {}".format(round(time.time()-start, 5)))
      self.train_loss = np.einsum('ij,ij->i',self.performance_matrix, self.series_models_map)/self.n_top
      self.train_loss = np.mean(self.train_loss)
      print("Training loss ({} prc): {:.5f}".format(self.mc.training_percentile,
                                                    self.train_loss))
      print('Models num series', np.sum(self.series_models_map, axis=0))

      if (epoch % self.mc.freq_of_test == 0) and (self.mc.freq_of_test > 0):
        if self.y_test_df is not None:
          self.evaluate_model_prediction(self.y_train_df, self.X_test_df,
                                        self.y_test_df, epoch=epoch)
    print('Train finished! \n')

  def predict(self, X_df):
    """
        Predictions for all stored time series
    Returns:
        Y_hat_panel : array-like (n_samples, 1).
          Predicted values for models in Family for ids in Panel.
        ds: Corresponding list of date stamps
        unique_id: Corresponding list of unique_id
    """
    assert type(X_df) == pd.core.frame.DataFrame
    assert 'unique_id' in X_df
    assert self._fitted, "Model not fitted yet"

    dataloader = Iterator(mc=self.mc, X=self.X, y=self.y)

    output_size = self.mc.output_size
    n_unique_id = len(dataloader.sort_key['unique_id'])

    ensemble_y_hat = np.zeros((self.n_models, n_unique_id, output_size))

    for model_id, esrnn in enumerate(self.esrnn_ensemble):
      esrnn.esrnn.eval()

      # Predict ALL series
      count = 0
      for j in range(dataloader.n_batches):
        batch = dataloader.get_batch()
        batch_size = batch.y.shape[0]

        y_hat = esrnn.esrnn.predict(batch)

        y_hat = y_hat.data.cpu().numpy()

        ensemble_y_hat[model_id, count:count+batch_size, :] = y_hat
        count += batch_size

    # Weighted average of prediction for n_top best models per series
    # (n_models x n_unique_id x output_size) (n_unique_id x n_models)
    y_hat = np.einsum('ijk,ji->jk', ensemble_y_hat, self.series_models_map) / self.n_top
    y_hat = y_hat.flatten()

    panel_unique_id = pd.Series(dataloader.sort_key['unique_id']).repeat(output_size)
    panel_last_ds = pd.Series(dataloader.X[:, 2]).repeat(output_size)

    panel_delta = list(range(1, output_size+1)) * n_unique_id
    panel_delta = pd.to_timedelta(panel_delta, unit=self.mc.frequency)
    panel_ds = panel_last_ds + panel_delta

    assert len(panel_ds) == len(y_hat) == len(panel_unique_id)

    Y_hat_panel_dict = {'unique_id': panel_unique_id,
                        'ds': panel_ds,
                        'y_hat': y_hat}

    Y_hat_panel = pd.DataFrame.from_dict(Y_hat_panel_dict)

    if 'ds' in X_df:
      Y_hat_panel = X_df.merge(Y_hat_panel, on=['unique_id', 'ds'], how='left')
    else:
      Y_hat_panel = X_df.merge(Y_hat_panel, on=['unique_id'], how='left')

    return Y_hat_panel

  def evaluate_model_prediction(self, y_train_df, X_test_df, y_test_df, epoch=None):
    """
    y_train_df: pandas df
      panel with columns unique_id, ds, y
    X_test_df: pandas df
      panel with columns unique_id, ds, x
    y_test_df: pandas df
      panel with columns unique_id, ds, y, y_hat_naive2
    model: python class
      python class with predict method
    """
    assert self._fitted, "Model not fitted yet"

    y_panel = y_test_df.filter(['unique_id', 'ds', 'y'])
    y_naive2_panel = y_test_df.filter(['unique_id', 'ds', 'y_hat_naive2'])
    y_naive2_panel.rename(columns={'y_hat_naive2': 'y_hat'}, inplace=True)
    y_hat_panel = self.predict(X_test_df)
    y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

    model_owa, model_mase, model_smape = owa(y_panel, y_hat_panel,
                                             y_naive2_panel, y_insample,
                                             seasonality=self.mc.naive_seasonality)

    if self.min_owa > model_owa:
      self.min_owa = model_owa
      if epoch is not None:
        self.min_epoch = epoch

    print('OWA: {} '.format(np.round(model_owa, 3)))
    print('SMAPE: {} '.format(np.round(model_smape, 3)))
    print('MASE: {} '.format(np.round(model_mase, 3)))

    return model_owa, model_mase, model_smape
