import os
import yaml
import argparse
import itertools
import ast
import pickle

import pandas as pd
import numpy as np

from src.M4_data import prepare_M4_data
from src.utils_evaluation import owa


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

#############################################################################
# HYPER PARAMETER GRIDS
#############################################################################


QUARTERLY = {'model_type': ['esrnn'],
             'dataset': ['Quarterly'],
             'max_epochs' : [10, 15],
             'batch_size' : [8, 32],
             'learning_rate' : [1e-4, 1e-3],
             'lr_scheduler_step_size' : [10],
             'per_series_lr_multip' : [1.0, 1.5],
             'gradient_clipping_threshold' : [20],
             'rnn_weight_decay' : [0.0, 0.5],
             'noise_std' : [1e-2],
             'level_variability_penalty' : [80, 100],
             'percentile' : [50],
             'training_percentile' : [45, 50],
             'max_periods': [10, 20],
             'state_hsize' : [40],
             'dilations' : [[[1, 2], [4, 8]], [[1,2,4,8]]],
             'add_nl_layer' : [True, False],
             'seasonality' : [4],
             'output_size' : [8],
             'device' : ['cuda']} #cuda:int

DAILY = {}

YEARLY = {}

HOURLY = {}

ALL_MODEL_SPEC  = {'Quarterly': QUARTERLY,
                   'Daily': DAILY,
                   'Yearly': YEARLY,
                   'Hourly': HOURLY}

#############################################################################
# MAIN EXPERIMENT
#############################################################################

def generate_grid(args, grid_file):
  model_specs = ALL_MODEL_SPEC[args.dataset]

  specs_list = list(itertools.product(*list(model_specs.values())))
  model_specs_df = pd.DataFrame(specs_list,
                            columns=list(model_specs.keys()))

  model_specs_df['model_id'] = model_specs_df.index
  np.random.seed(1)
  model_specs_df = model_specs_df.sample(100)
  model_specs_df.to_csv(grid_file, encoding='utf-8', index=None)

def grid_main(args):
  # Read train and test data
  X_df_train, y_df_train, X_df_test, y_df_test = prepare_M4_data(args.dataset, num_obs=1000)

  # Read/Generate hyperparameter grid
  grid_dir = './results/grid_search/{}/'.format(args.dataset)
  grid_file = grid_dir + '/model_grid.csv'
  if not os.path.exists(grid_dir):
    os.mkdir('.results/grid_search/')
    os.mkdir(grid_dir)
  if not os.path.exists(grid_file):
    generate_grid(args, grid_file)
  model_specs_df = pd.read_csv(grid_file)
  
  # Parse hyper parameter data frame
  for i in range(args.id_min, args.id_max):
    mc = model_specs_df.loc[i, :]
    
    dilations = ast.literal_eval(mc.dilations)
    device = mc.device

    print(47*'=' + '\n')
    print('model_config: {}'.format(i))
    print(mc)
    print('\n')
    print(47*'=' + '\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    from src.ESRNN import ESRNN

    # Instantiate, fit, predict and evaluate
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

    # Fit, predict and evaluate
    model.fit(X_df_train, y_df_train)
    model_owa = evaluate_model_prediction(y_df_train, X_df_test, y_df_test, model=model)
    evaluation_dict = {'id': mc.model_id, 'test evaluation': model_owa}

    # Output evaluation
    grid_dir = './results/grid_search/{}/'.format(args.dataset)
    output_file = '{}/model_{}.p'.format(grid_dir, mc.model_id)
    outfile = open(output_file, "wb")
    pickle.dump(evaluation_dict, outfile)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parser')
  parser.add_argument("--id_min", required=True, type=int)
  parser.add_argument("--id_max", required=True, type=int)
  parser.add_argument("--gpu_id", required=False, type=int, default=0)
  parser.add_argument("--dataset", required=True, type=str)
  args = parser.parse_args()

  grid_main(args)
