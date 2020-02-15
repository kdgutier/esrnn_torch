import os
from six.moves import urllib

import numpy as np
import pandas as pd

from src.utils_evaluation import Naive2

SOURCE_URL = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/'
DATA_DIRECTORY = "./data/m4/"
TRAIN_DIRECTORY = DATA_DIRECTORY + "Train/"
TEST_DIRECTORY = DATA_DIRECTORY + "Test/"


def maybe_download(filename):
  """Download the data from M4's website, unless it's already here."""
  if not os.path.exists(DATA_DIRECTORY):
    os.mkdir(DATA_DIRECTORY)
  if not os.path.exists(TRAIN_DIRECTORY):
    os.mkdir(TRAIN_DIRECTORY)
  if not os.path.exists(TEST_DIRECTORY):
    os.mkdir(TEST_DIRECTORY)
  
  filepath = os.path.join(DATA_DIRECTORY, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    size = os.path.getsize(filepath)
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def M4_parser(dataset_name, num_obs=1000000):
  m4_info = pd.read_csv(DATA_DIRECTORY+'/M4-info.csv', usecols=['M4id','category'])
  m4_info = m4_info[m4_info['M4id'].str.startswith(dataset_name[0])].reset_index(drop=True)
  
  # train data
  train_path='{}/train/{}-train.csv'.format(DATA_DIRECTORY, dataset_name)
  dataset = pd.read_csv(train_path).head(num_obs)
  dataset = dataset.rename(columns={'V1':'unique_id'})

  dataset = pd.wide_to_long(dataset, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset = dataset.rename(columns={'V':'y'})
  dataset = dataset.dropna()
  dataset.loc[:,'ds'] = pd.to_datetime(dataset['ds']-1, unit='d')
  dataset = dataset.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  dataset.drop(columns=['M4id'], inplace=True)
  dataset = dataset.rename(columns={'category': 'x'})
  dataset.sort_values(by=['unique_id', 'ds'], inplace=True)
  X_train_df = dataset.filter(items=['unique_id', 'ds', 'x'])
  y_train_df = dataset.filter(items=['unique_id', 'ds', 'y'])
  
  max_dates = X_train_df.groupby('unique_id').agg({'ds': 'count'}).reset_index()
  
  # test data
  test_path='{}/test/{}-test.csv'.format(DATA_DIRECTORY, dataset_name)
  dataset = pd.read_csv(test_path).head(num_obs)
  dataset = dataset.rename(columns={'V1':'unique_id'})

  dataset = pd.wide_to_long(dataset, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset = dataset.rename(columns={'V':'y'})
  dataset = dataset.dropna()

  dataset = dataset.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  dataset.drop(columns=['M4id'], inplace=True)
  dataset = dataset.rename(columns={'category': 'x'})
  
  dataset = dataset.merge(max_dates, on='unique_id', how='left')
  dataset['ds'] = dataset['ds_x'] + dataset['ds_y']
  dataset.loc[:,'ds'] = pd.to_datetime(dataset['ds']-1, unit='d')
  
  X_test_df = dataset.filter(items=['unique_id', 'x', 'ds'])
  y_test_df = dataset.filter(items=['unique_id', 'y', 'ds'])

  return X_train_df, y_train_df, X_test_df, y_test_df

def naive2_predictions(dataset_name, num_obs):
    print(9*'='+' Predicting Naive2 ' + 8*'=')

    # Read train and test data
    _, y_train_df, _, y_test_df = M4_parser(dataset_name, num_obs)
    
    seas_dict = {'Quarterly': {'seasonality': 4, 'input_size': 4,
                               'output_size': 8},
                 'Monthly': {'seasonality': 12, 'input_size': 12,
                             'output_size':24},
                 'Daily': {'seasonality': 7, 'input_size': 7,
                           'output_size': 14}}
    
    seasonality = seas_dict[dataset_name]['seasonality']
    input_size = seas_dict[dataset_name]['input_size']
    output_size = seas_dict[dataset_name]['output_size']

    print('Dataset:{} Seasonality:{}'.format(dataset_name, seasonality)+ '\n')
    
    # Naive2
    y_naive2_df = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])
    
    # Sort X by unique_id for faster loop
    y_train_df = y_train_df.sort_values(by=['unique_id', 'ds'])

    # List of uniques ids
    unique_ids = y_train_df['unique_id'].unique()

    # Panel of fitted models
    for unique_id in unique_ids:
        # Fast filter X and y by id.
        top_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'left'))
        bottom_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'right'))
        y_id = y_train_df[top_row:bottom_row]
        
        y_naive2 = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])
        y_naive2['ds'] = pd.date_range(start=y_id.ds.max(),
                                       periods=output_size+1, freq='D')[1:]
        y_naive2['unique_id'] = unique_id
        y_naive2['y_hat'] = Naive2(seasonality).fit(y_id.y.to_numpy()).predict(output_size)
        y_naive2_df = y_naive2_df.append(y_naive2)
    
    y_naive2_df = y_test_df.merge(y_naive2_df, on=['unique_id', 'ds'], how='left')
    y_naive2_df.rename(columns={'y_hat': 'y_hat_naive2'}, inplace=True)
    naive2_file = './results/{}-naive2predictions_{}.csv'.format(dataset_name, num_obs)
    y_naive2_df.to_csv(naive2_file, encoding='utf-8', index=None)
    return y_naive2_df

def prepare_M4_data(dataset_name, num_obs):
  m4info_filename = maybe_download('M4-info.csv')
  
  dailytrain_filename = maybe_download('Train/Daily-train.csv')
  hourlytrain_filename = maybe_download('Train/Hourly-train.csv')
  monthlytrain_filename = maybe_download('Train/Monthly-train.csv')
  quarterlytrain_filename = maybe_download('Train/Quarterly-train.csv')
  weeklytrain_filename = maybe_download('Train/Weekly-train.csv')
  yearlytrain_filename = maybe_download('Train/Yearly-train.csv')

  dailytest_filename = maybe_download('Test/Daily-test.csv')
  hourlytest_filename = maybe_download('Test/Hourly-test.csv')
  monthlytest_filename = maybe_download('Test/Monthly-test.csv')
  quarterlytest_filename = maybe_download('Test/Quarterly-test.csv')
  weeklytest_filename = maybe_download('Test/Weekly-test.csv')
  yearlytest_filename = maybe_download('Test/Yearly-test.csv')

  X_train_df, y_train_df, X_test_df, y_test_df = M4_parser(dataset_name, num_obs)

  naive2_file = './results/{}-naive2predictions_{}.csv'.format(dataset_name, num_obs)
  if not os.path.exists(naive2_file):
    y_naive2_df = naive2_predictions(dataset_name, num_obs)

  else:
    y_naive2_df = pd.read_csv(naive2_file)

  return X_train_df, y_train_df, X_test_df, y_naive2_df
