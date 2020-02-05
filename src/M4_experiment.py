import yaml

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



def main():
  X_df_train, y_df_train = M4_parser(dataset_name='Quarterly', mode='train', num_obs=1000)
  #X_df_test, y_df_test = M4_parser(dataset_name='Quarterly', mode='test')

  config_file = './configs/config_m4quarterly.yaml'
  with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

  esrnn = ESRNN(max_epochs=config['train_parameters']['max_epochs'],
                batch_size=config['train_parameters']['batch_size'],
                learning_rate=config['train_parameters']['learning_rate'],
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


if __name__ == '__main__':
  main()
