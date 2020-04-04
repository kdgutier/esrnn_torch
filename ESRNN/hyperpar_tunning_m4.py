import os
import yaml
import argparse
import itertools
import ast
import pickle

import numpy as np
import pandas as pd

from ESRNN.M4_data import prepare_M4_data
from ESRNN.utils_evaluation import owa
from ESRNN.utils_visualization import plot_cat_distributions

from statsmodels.formula.api import ols


#############################################################################
# HYPER PARAMETER GRIDS
#############################################################################

HOURLY = {'model_type': ['esrnn'],
          'dataset': ['Hourly'],
          'n_models': [5],
          'n_top': [4],
          'max_epochs' : [30],
          'batch_size' : [16, 32],
          'freq_of_test': [5],
          'learning_rate' : [1e-3, 1e-2],
          'lr_scheduler_step_size' : [7],
          'lr_decay' : [0.5],
          'per_series_lr_multip' : [1.0, 1.5],
          'gradient_clipping_threshold' : [50],
          'rnn_weight_decay' : [0.0, 0.05],
          'noise_std' : [1e-3],
          'level_variability_penalty' : [10, 30],
          'testing_percentile' : [50],
          'training_percentile' : [49, 50],
          'max_periods': [371],
          'cell_type': ['LSTM', 'ResLSTM'],
          'state_hsize' : [40],
          'dilations' : [[[1, 4], [24, 168]], [[1, 4, 24, 168]]],
          'add_nl_layer' : [True, False],
          'seasonality' : [[24, 168]],
          'input_size' : [24],
          'output_size' : [48],
          'random_seed': [1],
          'device' : ['cuda']}

DAILY = {'model_type': ['esrnn'],
         'dataset': ['Daily'],
         'n_models': [5],
         'n_top': [4],
         'max_epochs' : [20],
         'batch_size' : [64],
         'freq_of_test': [2],
         'learning_rate': [1e-2, 2e-2, 5e-2],
         'lr_scheduler_step_size' : [4],
         'lr_decay' : [0.333],
         'per_series_lr_multip': [0.5, 1.0, 1.5, 2.0, 3.0],
         'gradient_clipping_threshold' : [50],
         'rnn_weight_decay' : [0.00],
         'noise_std' : [1e-4],
         'level_variability_penalty' : [50, 100],
         'testing_percentile' : [50],
         'training_percentile' : [65, 62, 60, 55],
         'ensemble': [False],
         'max_periods': [15],
         'cell_type': ['LSTM', 'ResLSTM', 'AttentiveLSTM'],
         'state_hsize' : [40],
         'dilations' : [[[1, 7, 28]]],
         'add_nl_layer' : [True],
         'seasonality' : [[7]],
         'input_size' : [7],
         'output_size' : [14],
         'random_seed': [1],
         'device' : ['cuda']}

WEEKLY = {'model_type': ['esrnn'],
          'dataset': ['Weekly'],
          'n_models': [5],
          'n_top': [4],
          'max_epochs' : [30],
          'batch_size' : [16, 32, 64],
          'freq_of_test': [5],
          'learning_rate': [5e-4, 1e-3, 1.5e-3],
          'lr_scheduler_step_size' : [10, 15],
          'lr_decay' : [0.5],
          'per_series_lr_multip': [1.0, 2.0, 3.0],
          'gradient_clipping_threshold' : [20],
          'rnn_weight_decay' : [0.0],
          'noise_std' : [1e-3],
          'level_variability_penalty' : [100],
          'testing_percentile' : [50],
          'training_percentile' : [45, 48, 50],
          'ensemble': [True],
          'max_periods': [31],
          'cell_type': ['LSTM'],
          'state_hsize' : [40],
          'dilations' : [[[1, 52]]],
          'add_nl_layer' : [False],
          'seasonality' : [[]],
          'input_size' : [10],
          'output_size' : [13],
          'random_seed': [1, 2, 117, 120652, 117982, 1210357],
          'device' : ['cuda']}

MONTHLY = {'model_type': ['esrnn'],
           'dataset': ['Monthly'],
           'n_models': [5],
           'n_top': [4],
           'max_epochs' : [15, 25],
           'batch_size' : [8, 32, 128],
           'freq_of_test': [4],
           'learning_rate' : [5e-4, 6e-4, 7e-4],
           'lr_scheduler_step_size' : [12],
           'lr_decay' : [0.2],
           'per_series_lr_multip' : [0.5, 0.6, 0.7],
           'gradient_clipping_threshold' : [20],
           'rnn_weight_decay': [0.0],
           'noise_std' : [1e-3],
           'level_variability_penalty' : [50, 70, 80],
           'testing_percentile' : [50],
           'training_percentile' : [45, 50],
           'ensemble': [True, False],
           'max_periods': [36, 72],
           'cell_type': ['LSTM', 'ResLSTM'],
           'state_hsize' : [40, 50],
           'dilations' : [[[1,3],[6, 12]], [[1,3,6,12]]],
           'add_nl_layer' : [False, True],
           'seasonality' : [[12]],
           'input_size' : [12],
           'output_size' : [18],
           'random_seed': [1, 2, 3, 4, 5],
           'device' : ['cuda']}

QUARTERLY = {'model_type': ['esrnn'],
             'dataset': ['Quarterly'],
             'n_models': [5],
             'n_top': [4],
             'max_epochs' : [30],
             'batch_size' : [8, 16],
             'freq_of_test': [5],
             'learning_rate': [5e-4, 7e-4],
             'lr_scheduler_step_size' : [10],
             'lr_decay' : [0.1, 0.5],
             'per_series_lr_multip': [1.0, 2.0, 2.5, 3.0],
             'gradient_clipping_threshold' : [20],
             'rnn_weight_decay' : [0.0],
             'noise_std' : [1e-3],
             'level_variability_penalty' : [80, 100, 120],
             'testing_percentile' : [50],
             'training_percentile' : [50],
             'ensemble': [False, True],
             'max_periods': [12, 16, 20],
             'cell_type': ['LSTM'],
             'state_hsize' : [40],
             'dilations' : [[[1, 2], [4, 8]], [[1, 2, 4, 8]]],
             'add_nl_layer' : [False, True],
             'seasonality' : [[4]],
             'input_size' : [4],
             'output_size' : [8],
             'random_seed': [1, 2, 3, 4, 5],
             'device' : ['cuda']} #cuda:int

YEARLY = {'model_type': ['esrnn'],
          'dataset': ['Yearly'],
          'n_models': [5],
          'n_top': [4],
          'max_epochs' : [20, 40],
          'batch_size' : [4, 8, 16],
          'freq_of_test': [5],
          'learning_rate' : [3e-4],
          'lr_scheduler_step_size' : [10],
          'lr_decay' : [0.1],
          'per_series_lr_multip' : [0.8, 1.5, 2.0, 3.0],
          'gradient_clipping_threshold' : [50],
          'rnn_weight_decay' : [0.0],
          'noise_std' : [1e-3],
          'level_variability_penalty' : [0, 100],
          'testing_percentile' : [50],
          'training_percentile' : [50],
          'ensemble': [True],
          'max_periods': [25],
          'cell_type': ['LSTM','ResLSTM','AttentiveLSTM'],
          'state_hsize' : [30],
          'dilations' : [[[1], [6]]],
          'add_nl_layer' : [False],
          'seasonality' : [[]],
          'input_size' : [4],
          'output_size' : [6],
          'random_seed': [3,120652,117982,117,1210357, 123456],
          'device' : ['cuda']}

ALL_MODEL_SPECS  = {'Hourly': HOURLY,
                    'Daily': DAILY,
                    'Weekly': WEEKLY,
                    'Monthly': MONTHLY,
                    'Quarterly': QUARTERLY,
                    'Yearly': YEARLY}

#############################################################################
# MAIN EXPERIMENT
#############################################################################

def generate_grid(args, grid_file):
  model_specs = ALL_MODEL_SPECS[args.dataset]

  specs_list = list(itertools.product(*list(model_specs.values())))
  model_specs_df = pd.DataFrame(specs_list,
                            columns=list(model_specs.keys()))

  model_specs_df['model_id'] = model_specs_df.index
  np.random.seed(1)
  print('model_specs_df', model_specs_df)
  model_specs_df = model_specs_df.sample(200)
  model_specs_df.to_csv(grid_file, encoding='utf-8', index=None)

def grid_main(args):
  # Read train and test data
  X_train_df, y_train_df, X_test_df, y_test_df = prepare_M4_data(args.dataset, num_obs=1000000)

  # Read/Generate hyperparameter grid
  grid_dir = './results/grid_search/{}/'.format(args.dataset)
  grid_file = grid_dir + '/model_grid.csv'
  if not os.path.exists(grid_dir):
    if not os.path.exists('./results/grid_search/'):
      os.mkdir('./results/grid_search/')
    os.mkdir(grid_dir)
  if (not os.path.exists(grid_file)) or (args.gen_grid == 1):
    generate_grid(args, grid_file)
  model_specs_df = pd.read_csv(grid_file)

  # Parse hyper parameter data frame
  for i in range(args.id_min, args.id_max):
    mc = model_specs_df.loc[i, :]

    dilations = ast.literal_eval(mc.dilations)
    seasonality = ast.literal_eval(mc.seasonality)
    device = mc.device

    print(47*'=' + '\n')
    print('model_config: {}'.format(i))
    print(mc)
    print('\n')
    print(47*'=' + '\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    from src.ESRNNensemble import ESRNNensemble
    from src.ESRNN import ESRNN

    # Instantiate, fit, predict and evaluate
    model = ESRNNensemble(n_models=int(mc.n_models),
                          n_top=int(mc.n_top),
    #model = ESRNN(
                          max_epochs=int(mc.max_epochs),
                          batch_size=int(mc.batch_size),
                          learning_rate=mc.learning_rate,
                          lr_scheduler_step_size=mc.lr_scheduler_step_size,
                          per_series_lr_multip=mc.per_series_lr_multip,
                          gradient_clipping_threshold=mc.gradient_clipping_threshold,
                          rnn_weight_decay=mc.rnn_weight_decay,
                          noise_std=mc.noise_std,
                          level_variability_penalty=int(mc.level_variability_penalty),
                          testing_percentile=int(mc.testing_percentile),
                          training_percentile=int(mc.training_percentile),
                          cell_type=mc.cell_type,
                          ensemble=mc.cell_type,
                          state_hsize=int(mc.state_hsize),
                          dilations=dilations,
                          add_nl_layer=mc.add_nl_layer,
                          seasonality=seasonality,
                          input_size=int(mc.input_size),
                          output_size=int(mc.output_size),
                          freq_of_test=int(mc.freq_of_test),
                          random_seed=int(mc.random_seed),
                          device=device)

    # Fit, predict and evaluate
    model.fit(X_train_df, y_train_df, X_test_df, y_test_df)
    final_owa, final_mase, final_smape = model.evaluate_model_prediction(y_train_df,
                                                                         X_test_df,
                                                                         y_test_df)
    evaluation_dict = {'id': mc.model_id,
                       'min_owa': model.min_owa,
                       'min_epoch': model.min_epoch,
                       'owa': final_owa,
                       'mase': final_mase,
                       'smape': final_smape}

    # Output evaluation
    grid_dir = './results/grid_search/{}/'.format(args.dataset)
    output_file = '{}/model_{}.p'.format(grid_dir, mc.model_id)

    with open(output_file, "wb") as f:
      pickle.dump(evaluation_dict, f)

def parse_grid_search(dataset_name):
  gs_directory = './results/grid_search/{}/'.format(dataset_name)
  gs_file = gs_directory+'model_grid.csv'
  gs_df = pd.read_csv(gs_directory+'model_grid.csv', dtype=str)

  gs_df['min_owa'] = 0
  gs_df['min_epoch'] = 0
  gs_df['mase'] = 0
  gs_df['smape'] = 0

  results = []
  files = os.listdir(gs_directory)
  files.remove('model_grid.csv')

  for idx, row in gs_df.iterrows():
      file = gs_directory + 'model_' + str(row.model_id) + '.p'

      try:
        with open(file, 'rb') as pickle_file:
            results = pickle.load(pickle_file)
        gs_df.loc[idx, 'min_owa'] = results['min_owa']
        gs_df.loc[idx, 'min_epoch'] = results['min_epoch']
        gs_df.loc[idx, 'mase'] = results['mase']
        gs_df.loc[idx, 'smape'] = results['smape']
        gs_df.loc[idx, 'owa'] = results['owa']
      except:
        gs_df.loc[idx, 'min_owa'] = np.nan
        gs_df.loc[idx, 'min_epoch'] = np.nan
        gs_df.loc[idx, 'mase'] = np.nan
        gs_df.loc[idx, 'smape'] = np.nan
        gs_df.loc[idx, 'owa'] = np.nan

  # results = ols('min_owa ~ \
  #               learning_rate + per_series_lr_multip + batch_size + \
  #               dilations + ensemble + max_periods + \
  #               training_percentile + level_variability_penalty + state_hsize + \
  #               random_seed', data=gs_df).fit()
  #print(results.summary())

  return gs_df

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parser')
  parser.add_argument("--id_min", required=True, type=int)
  parser.add_argument("--id_max", required=True, type=int)
  parser.add_argument("--gpu_id", required=False, type=int, default=0)
  parser.add_argument("--dataset", required=True, type=str)
  parser.add_argument("--gen_grid", required=True, type=int)
  args = parser.parse_args()

  if args.dataset=='All':
    for dataset in ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']:
      args.dataset = dataset
      grid_main(args)
  if args.dataset=='Other':
    for dataset in ['Hourly', 'Daily', 'Weekly']:
      args.dataset = dataset
      grid_main(args)
  else:
    grid_main(args)
