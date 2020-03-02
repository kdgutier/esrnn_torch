import os
import yaml
import argparse
import itertools
import ast
import pickle

import pandas as pd
import numpy as np

from ESRNN.utils_evaluation import owa

def generate_grid(args):
  model_specs = {'model_type': ['esrnn'],
                 'dataset': [args.dataset],
                 'max_epochs' : [50, 500],
                 'batch_size' : [8, 16],
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
                 'dilations' : [[[1, 7], [28]], [[1, 7, 28]], [[1, 28, 84]]],
                 'add_nl_layer' : [True, False],
                 'seasonality' : [7, 28],
                 'output_size' : [34],
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
  X_train = pd.read_csv('./data/' + args.dataset + '/X_train.csv')
  y_train = pd.read_csv('./data/' + args.dataset + '/y_train.csv')
  X_test = pd.read_csv('./data/' + args.dataset + '/X_test.csv')
  
  X_train['ds'] = pd.to_datetime(X_train['ds'])
  X_test['ds'] = pd.to_datetime(X_test['ds'])

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

