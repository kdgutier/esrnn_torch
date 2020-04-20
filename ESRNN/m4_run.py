import os
import argparse
import itertools
import ast
import pickle
import time

import os
import numpy as np
import pandas as pd

from ESRNN.m4_data import prepare_m4_data
from ESRNN.utils_evaluation import evaluate_prediction_owa
from ESRNN.utils_configs import get_config

from ESRNN import ESRNN

import torch

def main(args):
  config = get_config(args.dataset)
  if config['data_parameters']['frequency'] == 'Y':
      config['data_parameters']['frequency'] = None

  #Setting needed parameters
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

  if args.num_obs:
      num_obs = args.num_obs
  else:
      num_obs = 100000

  if args.use_cpu == 1:
      config['device'] = 'cpu'
  else:
      assert torch.cuda.is_available(), 'No cuda devices detected. You can try using CPU instead.'

  #Reading data
  print('Reading data')
  X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=args.dataset,
                                                                 directory=args.results_directory,
                                                                 num_obs=num_obs)

  # Instantiate model
  model = ESRNN(max_epochs=config['train_parameters']['max_epochs'],
                batch_size=config['train_parameters']['batch_size'],
                freq_of_test=config['train_parameters']['freq_of_test'],
                learning_rate=float(config['train_parameters']['learning_rate']),
                lr_scheduler_step_size=config['train_parameters']['lr_scheduler_step_size'],
                lr_decay=config['train_parameters']['lr_decay'],
                per_series_lr_multip=config['train_parameters']['per_series_lr_multip'],
                gradient_clipping_threshold=config['train_parameters']['gradient_clipping_threshold'],
                rnn_weight_decay=config['train_parameters']['rnn_weight_decay'],
                noise_std=config['train_parameters']['noise_std'],
                level_variability_penalty=config['train_parameters']['level_variability_penalty'],
                testing_percentile=config['train_parameters']['testing_percentile'],
                training_percentile=config['train_parameters']['training_percentile'],
                ensemble=config['train_parameters']['ensemble'],
                max_periods=config['data_parameters']['max_periods'],
                seasonality=config['data_parameters']['seasonality'],
                input_size=config['data_parameters']['input_size'],
                output_size=config['data_parameters']['output_size'],
                frequency=config['data_parameters']['frequency'],
                cell_type=config['model_parameters']['cell_type'],
                state_hsize=config['model_parameters']['state_hsize'],
                dilations=config['model_parameters']['dilations'],
                add_nl_layer=config['model_parameters']['add_nl_layer'],
                random_seed=config['model_parameters']['random_seed'],
                device=config['device'])

  if args.test == 1:
       model = ESRNN(max_epochs=1,
                     batch_size=20,
                     seasonality=config['data_parameters']['seasonality'],
                     input_size=config['data_parameters']['input_size'],
                     output_size=config['data_parameters']['output_size'],
                     frequency=config['data_parameters']['frequency'],
                     device=config['device'])

  # Fit model
  # If y_test_df is provided the model will evaluate predictions on this set every freq_test epochs
  model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

  # Predict on test set
  print('\nForecasting')
  y_hat_df = model.predict(X_test_df)

  # Evaluate predictions
  print(15*'=', ' Final evaluation ', 14*'=')
  seasonality = config['data_parameters']['seasonality']
  if not seasonality:
      seasonality = 1
  else:
      seasonality = seasonality[0]

  final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train_df,
                                                               X_test_df, y_test_df,
                                                               naive2_seasonality=seasonality)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Replicate M4 results for the ESRNN model')
  parser.add_argument("--dataset", required=True, type=str,
                      choices=['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Hourly', 'Daily'],
                      help="set of M4 time series to be tested")
  parser.add_argument("--results_directory", required=True, type=str,
                      help="directory where M4 data will be downloaded")
  parser.add_argument("--gpu_id", required=False, type=int,
                      help="an integer that specify which GPU will be used")
  parser.add_argument("--use_cpu", required=False, type=int,
                      help="1 to use CPU instead of GPU (uses GPU by default)")
  parser.add_argument("--num_obs", required=False, type=int,
                      help="number of M4 time series to be tested (uses all data by default)")
  parser.add_argument("--test", required=False, type=int,
                      help="run fast for tests (no test by default)")
  args = parser.parse_args()

  main(args)
