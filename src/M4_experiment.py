import yaml
import argparse
import itertools
import ast
import pickle

import pandas as pd
import numpy as np

from src.ESRNN import ESRNN
from src.utils_evaluation import owa

def M4_parser(dataset_name, mode='train', num_obs=1000, data_dir='./data/m4'):
  m4_info = pd.read_csv(data_dir+'/M4-info.csv', usecols=['M4id','category'])
  m4_info = m4_info[m4_info['M4id'].str.startswith(dataset_name[0])].reset_index(drop=True)

  file_path='{}/{}/{}-{}.csv'.format(data_dir, mode, dataset_name, mode)
  dataset = pd.read_csv(file_path).head(num_obs)
  dataset = dataset.rename(columns={'V1':'unique_id'})
  
  dataset = pd.wide_to_long(dataset, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset = dataset.rename(columns={'V':'y'})
  dataset = dataset.dropna()
  dataset.loc[:,'ds'] = pd.to_datetime(dataset['ds']-1, unit='d')
  dataset = dataset.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  dataset.drop(columns=['M4id'], inplace=True)
  dataset = dataset.rename(columns={'category': 'x'})
  X_df = dataset.filter(items=['unique_id', 'ds', 'x'])
  y_df = dataset.filter(items=['unique_id', 'ds', 'y'])
  return X_df, y_df



def yaml_main():
  X_df_train, y_df_train = M4_parser(dataset_name='Quarterly', mode='train', num_obs=1000)
  X_df_test, y_df_test = M4_parser(dataset_name='Quarterly', mode='test')

  config_file = './configs/config_m4quarterly.yaml'
  with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

  esrnn = ESRNN(max_epochs=2,
                batch_size=config['train_parameters']['batch_size'],
                learning_rate=float(config['train_parameters']['learning_rate']),
                lr_scheduler_step_size=config['train_parameters']['lr_scheduler_step_size'],
                per_series_lr_multip=config['train_parameters']['per_series_lr_multip'],
                gradient_clipping_threshold=config['train_parameters']['gradient_clipping_threshold'],
                rnn_weight_decay=config['train_parameters']['rnn_weight_decay'],
                noise_std=config['train_parameters']['noise_std'],
                level_variability_penalty=config['train_parameters']['level_variability_penalty'],
                percentile=config['train_parameters']['percentile'],
                training_percentile=config['train_parameters']['training_percentile'],
                state_hsize=config['model_parameters']['state_hsize'],
                dilations=config['model_parameters']['dilations'],
                add_nl_layer=config['model_parameters']['add_nl_layer'],
                seasonality=config['data_parameters']['seasonality'],
                input_size=config['data_parameters']['input_size'],
                output_size=config['data_parameters']['output_size'],
                device=config['device'])

  esrnn.fit(X_df_train, y_df_train)
  y_df_hat = esrnn.predict(X_df_test)


def grid_main(args):
  model_specs = {'model_type': ['esrnn'],
                  'dataset': ['quarterly'],
                  'max_epochs' : [100],
                  'batch_size' : [2, 8, 32],
                  'learning_rate' : [1e-4, 1e-3, 1e-2],
                  'lr_scheduler_step_size' : [10],
                  'per_series_lr_multip' : [1.0, 0.5],
                  'gradient_clipping_threshold' : [20],
                  'rnn_weight_decay' : [0.0, 0.5],
                  'noise_std' : [1e-2],
                  'level_variability_penalty' : [40, 80, 100],
                  'percentile' : [50],
                  'training_percentile' : [49, 45],
                  'state_hsize' : [40],
                  'dilations' : [[[1, 2], [4, 8]]],
                  'add_nl_layer' : [True, False],
                  'seasonality' : [4],
                  'input_size' : [4],
                  'output_size' : [8],
                  'device' : ['cpu']} #cuda:int

  specs_list = list(itertools.product(*list(model_specs.values())))
  model_specs_df = pd.DataFrame(specs_list,
                              columns=list(model_specs.keys()))

  model_specs_df['model_id'] = model_specs_df.index
  grid_file = './data/quarterly_model_grid.csv'
  model_specs_df.to_csv(grid_file, encoding='utf-8', index=None)


  # Read train and test data
  X_df_train, y_df_train = M4_parser(dataset_name='Quarterly', mode='train', num_obs=1000)
  X_df_test, y_df_test = M4_parser(dataset_name='Quarterly', mode='test', num_obs=1000)

  model_specs_df = pd.read_csv(grid_file)
  mc = model_specs_df.loc[args.model_id, :]
  dilations = ast.literal_eval(mc.dilations)

  print(47*'=' + '\n')
  print('model_config: {}'.format(args.model_id))
  print(mc)
  print('\n')
  print(47*'=' + '\n')

  # Instantiate, fit and predict
  model = ESRNN(max_epochs=1,
                batch_size=mc.batch_size,
                learning_rate=mc.learning_rate,
                lr_scheduler_step_size=mc.lr_scheduler_step_size,
                per_series_lr_multip=mc.per_series_lr_multip,
                gradient_clipping_threshold=mc.gradient_clipping_threshold,
                rnn_weight_decay=mc.rnn_weight_decay,
                noise_std=mc.noise_std,
                level_variability_penalty=mc.level_variability_penalty,
                percentile=mc.percentile,
                training_percentile=mc.training_percentile,
                state_hsize=mc.state_hsize,
                dilations=dilations,
                add_nl_layer=mc.add_nl_layer,
                seasonality=mc.seasonality,
                input_size=mc.input_size,
                output_size=mc.output_size,
                device=mc.device)

  model.fit(X_df_train, y_df_train)
  y_df_hat = model.predict(X_df_test)

  # Dictionary generation
  evaluation_dict = {'train loss': model.train_loss,
                      'test evaluation': model.test_evaluation}

  # Output evaluation
  output_file = './results/grid_search/quarterly_model_{}.p'.format(args.model_id)
  outfile = open(output_file, "wb")
  pickle.dump(evaluation_dict, outfile)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parser')
  parser.add_argument("--model_id", required=True, type=int)
  args = parser.parse_args()

  grid_main(args)
