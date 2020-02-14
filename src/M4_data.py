import os
from six.moves import urllib
import pandas as pd

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

  X_train_df, y_train_df, X_test_df, y_test_df = M4_parser(dataset_name, 
                                                           num_obs=num_obs)

  return X_train_df, y_train_df, X_test_df, y_test_df
