import os
import yaml
import argparse
import itertools
import ast
import pickle

import pandas as pd
import numpy as np

from src.utils_evaluation import owa

def M4_parser(dataset_name, num_obs=1000000, data_dir='./data/m4'):
  m4_info = pd.read_csv(data_dir+'/M4-info.csv', usecols=['M4id','category'])
  m4_info = m4_info[m4_info['M4id'].str.startswith(dataset_name[0])].reset_index(drop=True)

  train_path='{}/train/{}-train.csv'.format(data_dir, dataset_name)
  dataset = pd.read_csv(train_path).head(num_obs)
  dataset = dataset.rename(columns={'V1':'unique_id'})
  
  dataset = pd.wide_to_long(dataset, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset = dataset.rename(columns={'V':'y'})
  dataset = dataset.dropna()
  dataset.loc[:,'ds'] = pd.to_datetime(dataset['ds']-1, unit='d')
  dataset = dataset.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  dataset.drop(columns=['M4id'], inplace=True)
  dataset = dataset.rename(columns={'category': 'x'})
  X_train_df = dataset.filter(items=['unique_id', 'ds', 'x'])
  y_train_df = dataset.filter(items=['unique_id', 'ds', 'y'])

  test_path='{}/test/{}-test.csv'.format(data_dir, dataset_name)
  dataset = pd.read_csv(test_path).head(num_obs)
  dataset = dataset.rename(columns={'V1':'unique_id'})
  
  dataset = pd.wide_to_long(dataset, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset = dataset.rename(columns={'V':'y'})
  dataset = dataset.dropna()

  dataset.loc[:,'ds'] = pd.to_datetime(dataset['ds']-1, unit='d')

  for unique_id in dataset.unique_id.unique():
      f_train_df = y_train_df.loc[y_train_df.unique_id==unique_id, :].reset_index()
      f_test_df = dataset.loc[dataset.unique_id==unique_id, :]

      ds = pd.date_range(start=f_train_df.ds.max(),
                         periods=len(f_test_df)+1, freq='D')[1:]

      dataset.loc[dataset.unique_id==unique_id, 'ds'] = ds

  dataset = dataset.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  dataset.drop(columns=['M4id'], inplace=True)
  dataset = dataset.rename(columns={'category': 'x'})
  X_test_df = dataset.filter(items=['unique_id', 'ds', 'x'])
  y_test_df = dataset.filter(items=['unique_id', 'ds', 'y'])

  return X_train_df, y_train_df, X_test_df, y_test_df



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

def naive2_predictions(dataset):
    # Read train and test data
    _, y_train_df, _, y_test_df = M4_parser(dataset_name=dataset)
    
    seas_dict = {'Quarterly': {'seasonality': 4, 'input_size': 4,
                               'output_size': 8},
                 'Monthly': {'seasonality': 12, 'input_size': 12,
                             'output_size':24},
                 'Daily': {'seasonality': 7, 'input_size': 7,
                           'output_size': 14}}
    
    seasonality = seas_dict[dataset]['seasonality']
    input_size = seas_dict[dataset]['input_size']
    output_size = seas_dict[dataset]['output_size']
    
    # Naive2
    y_naive2_df = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])
    
    for unique_id in y_train_df.unique_id.unique():
        y_df = y_train_df.loc[y_train_df.unique_id==unique_id, :].reset_index()
        y_naive2 = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])
        y_naive2['ds'] = pd.date_range(start=y_df.ds.max(),
                                       periods=output_size+1, freq='D')[1:]
        y_naive2['unique_id'] = [y_df.unique_id[0]] * output_size
        y_naive2['y_hat'] = Naive2(seasonality).fit(y_df.y.to_numpy()).predict(output_size)
        y_naive2_df = y_naive2_df.append(y_naive2)
    
    y_naive2_df['y'] = y_test_df['y']
    naive2_file = './results/{}/naive2predictions.csv'.format(dataset)
    y_naive2_df.to_csv(naive2_file, encoding='utf-8', index=None)

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
