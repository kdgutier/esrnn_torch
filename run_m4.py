import os
import yaml
import argparse
import itertools
import ast
import pickle
import time

import os
import numpy as np
import pandas as pd

from src.M4_data import prepare_M4_data
from src.utils_evaluation import evaluate_prediction_owa

def main(args):
  config_file = './configs/{}.yaml'.format(args.dataset)
  with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
  from src.ESRNN import ESRNN

  X_train_df, y_train_df, X_test_df, y_test_df = prepare_M4_data(dataset_name=args.dataset, num_obs=100000)
  
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
                percentile=config['train_parameters']['percentile'],
                training_percentile=config['train_parameters']['training_percentile'],
                ensemble=config['train_parameters']['ensemble'],
                max_periods=config['data_parameters']['max_periods'],
                seasonality=config['data_parameters']['seasonality'],
                input_size=config['data_parameters']['input_size'],
                output_size=config['data_parameters']['output_size'],
                cell_type=config['model_parameters']['cell_type'],
                state_hsize=config['model_parameters']['state_hsize'],
                dilations=config['model_parameters']['dilations'],
                add_nl_layer=config['model_parameters']['add_nl_layer'],
                random_seed=config['model_parameters']['random_seed'],
                device=config['device'])

  # Fit model
  # If y_test_df is provided the model will evaluate predictions on this set every freq_test epochs
  model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

  # Predict on test set
  y_hat_df = model.predict(X_test_df)

  # Evaluate predictions
  print(15*'=', ' Final evaluation ', 14*'=')
  final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train_df, 
                                                               X_test_df, y_test_df,
                                                               naive2_seasonality=config['data_parameters']['seasonality'][0])

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parser')
  parser.add_argument("--dataset", required=True, type=str)
  parser.add_argument("--gpu_id", required=False, type=int)
  args = parser.parse_args()

  main(args)
