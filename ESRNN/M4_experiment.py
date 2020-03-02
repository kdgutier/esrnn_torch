import os
import yaml
import argparse
import itertools
import ast
import pickle

import pandas as pd
import numpy as np

from ESRNN.M4_data import prepare_M4_data
from ESRNN.utils_evaluation import owa
from ESRNN.utils_visualization import plot_prediction

##############################################################################
#
##############################################################################

def plot_model_prediction(y_train_df, X_test_df, y_test_df, model, u_id):
    """
    y_train_df: pandas df
      panel with columns unique_id, ds, y
    X_test_df: pandas df
      panel with columns unique_id, ds, x
    y_test_df: pandas df
      panel with columns unique_id, ds, y
    model: python class
      python class with predict method
    u_id: str
      unique_id to filter pandas dfs
    """
    X_test_plot = X_test_df.loc[X_test_df.unique_id==u_id, :]
    y_train_plot = y_train_df.loc[y_train_df.unique_id==u_id, :]
    y_test_plot = y_test_df.loc[y_test_df.unique_id==u_id, :]
    y_test_plot = pd.concat([y_train_plot, y_test_plot], axis=0, sort=True)
    y_hat_plot = model.predict(X_test_plot)
    plot_prediction(y_test_plot, y_hat_plot)

def evaluate_model_prediction(y_train_df, X_test_df, y_test_df, model):
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
    y_panel = y_test_df.filter(['unique_id', 'ds', 'y'])
    y_naive2_panel = y_test_df.filter(['unique_id', 'ds', 'y_hat_naive2'])
    y_naive2_panel.rename(columns={'y_hat_naive2': 'y_hat'}, inplace=True)
    y_hat_panel = model.predict(X_test_df)
    y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

    model_owa = owa(y_panel, y_hat_panel, y_naive2_panel, y_insample,
                    seasonality=model.mc.seasonality)

    print('=='+' Overall Weighted Average:{} '.format(model_owa) + '==')
    return model_owa


##############################################################################
#
##############################################################################

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

def generate_grid(args):
  model_specs = {'model_type': ['esrnn'],
                 'dataset': [args.dataset],
                 'max_epochs' : [50, 500],
                 'batch_size' : [8, 16, 32],
                 'learning_rate' : [1e-4, 1e-3],
                 'lr_scheduler_step_size' : [10],
                 'per_series_lr_multip' : [1.0, 1.5],
                 'gradient_clipping_threshold' : [20],
                 'rnn_weight_decay' : [0.0, 0.5],
                 'noise_std' : [1e-2],
                 'level_variability_penalty' : [80, 100],
                 'percentile' : [50],
                 'training_percentile' : [45, 50],
                 'max_periods': [2, 20],
                 'state_hsize' : [40],
                 'dilations' : [[[1, 2], [4, 8]], [[1,2,4,8]]],
                 'add_nl_layer' : [True, False],
                 'seasonality' : [4],
                 'output_size' : [8],
                 'device' : ['cuda']} #cuda:int

  specs_list = list(itertools.product(*list(model_specs.values())))
  model_specs_df = pd.DataFrame(specs_list,
                            columns=list(model_specs.keys()))

  model_specs_df['model_id'] = model_specs_df.index
  grid_file = './data/' + args.dataset + '/model_grid.csv'
  np.random.seed(1)
  model_specs_df = model_specs_df.sample(100)
  model_specs_df.to_csv(grid_file, encoding='utf-8', index=None)

def grid_main(args):

  # Read train and test data
  X_df_train, y_df_train = M4_parser(dataset_name=args.dataset, mode='train')
  X_df_test, y_df_test = M4_parser(dataset_name=args.dataset, mode='test')

  grid_file = './data/' + args.dataset + '/model_grid.csv'
  model_specs_df = pd.read_csv(grid_file)

  for i in range(args.id_min, args.id_max):
    mc = model_specs_df.loc[i, :]

    dilations = ast.literal_eval(mc.dilations)
    device = mc.device# + ':' + str(args.gpu_id)

    print(47*'=' + '\n')
    print('model_config: {}'.format(i))
    print(mc)
    print('\n')
    print(47*'=' + '\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    from src.ESRNN import ESRNN

    # Instantiate, fit and predict
    model = ESRNN(max_epochs=int(mc.max_epochs),
                  batch_size=int(mc.batch_size),
                  learning_rate=mc.learning_rate,
                  lr_scheduler_step_size=mc.lr_scheduler_step_size,
                  per_series_lr_multip=mc.per_series_lr_multip,
                  gradient_clipping_threshold=mc.gradient_clipping_threshold,
                  rnn_weight_decay=mc.rnn_weight_decay,
                  noise_std=mc.noise_std,
                  level_variability_penalty=int(mc.level_variability_penalty),
                  percentile=int(mc.percentile),
                  training_percentile=int(mc.training_percentile),
                  state_hsize=int(mc.state_hsize),
                  dilations=dilations,
                  add_nl_layer=mc.add_nl_layer,
                  seasonality=int(mc.seasonality),
                  input_size=int(mc.seasonality),
                  output_size=int(mc.output_size),
                  freq_of_test=10000,
                  device=device)

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    loss = np.abs(y_hat['y_hat']-y_hat['y']).mean()

    # Dictionary generation
    evaluation_dict = {'id': mc.model_id,
                       'test evaluation': loss}

    # Output evaluation
    output_file = './results/grid_search/{}/{}.p'.format(args.dataset, mc.model_id)
    outfile = open(output_file, "wb")
    pickle.dump(evaluation_dict, outfile)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parser')
  parser.add_argument("--gen_grid", required=True, type=int)
  parser.add_argument("--id_min", required=True, type=int)
  parser.add_argument("--id_max", required=True, type=int)
  parser.add_argument("--gpu_id", required=True, type=int)
  parser.add_argument("--dataset", required=True, type=str)
  args = parser.parse_args()

  if args.gen_grid == 1:
      generate_grid(args)

  grid_main(args)
