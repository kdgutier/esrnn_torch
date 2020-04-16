import os
from six.moves import urllib
import subprocess

import numpy as np
import pandas as pd

from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from ESRNN.utils_evaluation import Naive2


seas_dict = {'Hourly': {'seasonality': 24, 'input_size': 24,
                       'output_size': 48, 'freq': 'H'},
             'Daily': {'seasonality': 7, 'input_size': 7,
                       'output_size': 14, 'freq': 'D'},
             'Weekly': {'seasonality': 52, 'input_size': 52,
                        'output_size': 13, 'freq': 'W'},
             'Monthly': {'seasonality': 12, 'input_size': 12,
                         'output_size':18, 'freq': 'M'},
             'Quarterly': {'seasonality': 4, 'input_size': 4,
                           'output_size': 8, 'freq': 'Q'},
             'Yearly': {'seasonality': 1, 'input_size': 4,
                        'output_size': 6, 'freq': 'Y'}}

SOURCE_URL = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/'


def maybe_download(filename, directory):
  """
  Download the data from M4's website, unless it's already here.

  Parameters
  ----------
  filename: str
    Filename of M4 data with format /Type/Frequency.csv. Example: /Test/Daily-train.csv
  directory: str
    Custom directory where data will be downloaded.
  """
  data_directory = directory + "/m4"
  train_directory = data_directory + "/Train/"
  test_directory = data_directory + "/Test/"

  if not os.path.exists(data_directory):
    os.mkdir(data_directory)
  if not os.path.exists(train_directory):
    os.mkdir(train_directory)
  if not os.path.exists(test_directory):
    os.mkdir(test_directory)

  filepath = os.path.join(data_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    size = os.path.getsize(filepath)
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def m4_parser(dataset_name, directory, num_obs=1000000):
  """
  Transform M4 data into a panel.

  Parameters
  ----------
  dataset_name: str
    Frequency of the data. Example: 'Yearly'.
  directory: str
    Custom directory where data will be saved.
  num_obs: int
    Number of time series to return.
  """
  data_directory = directory + "/m4"
  train_directory = data_directory + "/Train/"
  test_directory = data_directory + "/Test/"
  freq = seas_dict[dataset_name]['freq']

  # train data
  train_path='{}{}-train.csv'.format(train_directory, dataset_name)

  dataset = pd.read_csv(train_path).head(num_obs)
  dataset = dataset.rename(columns={'V1':'unique_id'})
  dataset = pd.wide_to_long(dataset, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset = dataset.rename(columns={'V':'y'})
  dataset = dataset.dropna()
  dataset['train'] = True

  max_dates = dataset.groupby('unique_id').agg({'ds': 'count'}).reset_index()

  # test data
  test_path='{}{}-test.csv'.format(test_directory, dataset_name)

  dataset_test = pd.read_csv(test_path).head(num_obs)
  dataset_test = dataset_test.rename(columns={'V1':'unique_id'})
  dataset_test = pd.wide_to_long(dataset_test, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset_test = dataset_test.rename(columns={'V':'y'})
  dataset_test = dataset_test.dropna()

  # Uniforming ds
  dataset_test = dataset_test.merge(max_dates, on='unique_id', how='left')
  dataset_test['ds'] = dataset_test['ds_x'] + dataset_test['ds_y']
  dataset_test = dataset_test.drop(columns=['ds_x', 'ds_y'])
  dataset_test['train'] = False

  # All data
  dataset = pd.concat([dataset, dataset_test], sort=True)
  del dataset_test

  dataset['ds'] -= 2
  dataset.sort_values(by=['unique_id', 'ds'], inplace=True)

  dataset = add_date(dataset, freq, data_directory=data_directory)

  X_train_df = dataset[dataset['train']].filter(items=['unique_id', 'ds', 'x'])
  y_train_df = dataset[dataset['train']].filter(items=['unique_id', 'ds', 'y'])

  X_test_df = dataset[~dataset['train']].filter(items=['unique_id', 'ds', 'x'])
  y_test_df = dataset[~dataset['train']].filter(items=['unique_id', 'ds', 'y'])

  return X_train_df, y_train_df, X_test_df, y_test_df

def naive2_predictions(dataset_name, directory, num_obs, y_train_df = None, y_test_df = None, py_predictions=True):
    """
    Computes Naive2 predictions.

    Parameters
    ----------
    dataset_name: str
      Frequency of the data. Example: 'Yearly'.
    directory: str
      Custom directory where data will be saved.
    num_obs: int
      Number of time series to return.
    y_train_df: DataFrame
      Y train set returned by m4_parser
    y_test_df: DataFrame
      Y test set returned by m4_parser
    """
    # Read train and test data
    if (y_train_df is None) or (y_test_df is None):
        _, y_train_df, _, y_test_df = m4_parser(dataset_name, directory, num_obs)

    seasonality = seas_dict[dataset_name]['seasonality']
    input_size = seas_dict[dataset_name]['input_size']
    output_size = seas_dict[dataset_name]['output_size']
    freq = seas_dict[dataset_name]['freq']

    print('Preparing {} dataset'.format(dataset_name))
    print('Preparing Naive2 {} dataset predictions'.format(dataset_name))

    # Sort X by unique_id for faster loop
    y_train_df = y_train_df.sort_values(by=['unique_id', 'ds'])

    if py_predictions:
        y_naive2_df = wrapper_naive2_python(y_train_df, y_test_df, seasonality, input_size, output_size, freq)
        identifier = ''
    else:
        y_naive2_df = wrapper_naive2_r(y_train_df, y_test_df, dataset_name, directory, num_obs, freq)
        identifier = '_r'

    y_naive2_df = add_date(y_naive2_df, freq, y_train_df=y_train_df)
    y_naive2_df = y_naive2_df.drop(columns='StartingDate').rename(columns={'y_hat': 'y_hat_naive2'})
    y_naive2_df = y_test_df.merge(y_naive2_df, on=['unique_id', 'ds'], how='left')

    results_dir = directory + '/results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    naive2_file = results_dir + '/{}-naive2predictions_{}{}.csv'.format(dataset_name, num_obs, identifier)
    y_naive2_df.to_csv(naive2_file, encoding='utf-8', index=None)

    return y_naive2_df

def prepare_m4_data(dataset_name, directory, num_obs, py_predictions=True):
  """
  Pipeline that obtains M4 times series, tranforms it and gets naive2 predictions.

  Parameters
  ----------
  dataset_name: str
    Frequency of the data. Example: 'Yearly'.
  directory: str
    Custom directory where data will be saved.
  num_obs: int
    Number of time series to return.
  py_predictions: bool
    whether use python or r predictions
  """
  m4info_filename = maybe_download('M4-info.csv', directory)

  dailytrain_filename = maybe_download('Train/Daily-train.csv', directory)
  hourlytrain_filename = maybe_download('Train/Hourly-train.csv', directory)
  monthlytrain_filename = maybe_download('Train/Monthly-train.csv', directory)
  quarterlytrain_filename = maybe_download('Train/Quarterly-train.csv', directory)
  weeklytrain_filename = maybe_download('Train/Weekly-train.csv', directory)
  yearlytrain_filename = maybe_download('Train/Yearly-train.csv', directory)

  dailytest_filename = maybe_download('Test/Daily-test.csv', directory)
  hourlytest_filename = maybe_download('Test/Hourly-test.csv', directory)
  monthlytest_filename = maybe_download('Test/Monthly-test.csv', directory)
  quarterlytest_filename = maybe_download('Test/Quarterly-test.csv', directory)
  weeklytest_filename = maybe_download('Test/Weekly-test.csv', directory)
  yearlytest_filename = maybe_download('Test/Yearly-test.csv', directory)
  print('\n')

  X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(dataset_name, directory, num_obs)

  results_dir = directory + '/results'
  if not os.path.exists(results_dir):
      os.mkdir(results_dir)

  naive2_file = results_dir + '/{}-naive2predictions_{}{}.csv'

  if py_predictions:
    naive2_file = naive2_file.format(dataset_name, num_obs, '')
  else:
    naive2_file = naive2_file.format(dataset_name, num_obs, '_r')

  if not os.path.exists(naive2_file):
    y_naive2_df = naive2_predictions(dataset_name, directory, num_obs, y_train_df, y_test_df, py_predictions)
  else:
    y_naive2_df = pd.read_csv(naive2_file)
    y_naive2_df['ds'] = pd.to_datetime(y_naive2_df['ds'])

  return X_train_df, y_train_df, X_test_df, y_naive2_df


#############################################################################
#### Naive2, python, R ######################################################
#############################################################################

def wrapper_naive2_python(y_train_df, y_test_df, seasonality, input_size, output_size, freq):
    # Naive2
    y_naive2_df = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])

    # List of uniques ids
    unique_ids = y_train_df['unique_id'].unique()

    #ds preds for all series
    ds_preds =  np.arange(1, output_size+1)

    # Panel of fitted models
    for unique_id in unique_ids:
        # Fast filter X and y by id.
        top_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'left'))
        bottom_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'right'))
        y_id = y_train_df[top_row:bottom_row]

        y_naive2 = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])
        y_naive2['ds'] = ds_preds

        #y_naive2['ds'] = y_naive2['StartingDate'] + y_naive2['ds'].apply(lambda x: custom_offset(freq, x, int_ds))
        #y_naive2.drop(columns='StartingDate', inplace=True)

        y_naive2['unique_id'] = unique_id
        y_naive2['y_hat'] = Naive2(seasonality).fit(y_id.y.to_numpy()).predict(output_size)
        y_naive2_df = y_naive2_df.append(y_naive2)

    return y_naive2_df

def wrapper_naive2_r(y_train_df, y_test_df, dataset_name, directory, num_obs, freq):

    str_call = 'Rscript ESRNN/r/Naive2.R --dataset_name {} --directory {} --num_obs {}'
    str_call = str_call.format(dataset_name, directory, num_obs)
    subprocess.call(str_call, shell=True)

    results_dir = directory + '/results'
    naive2_file = results_dir + '/{}-naive2predictions_{}{}.csv'
    naive2_file_raw = naive2_file.format(dataset_name, num_obs, '_r_raw')

    y_naive2_df = pd.read_csv(naive2_file_raw)
    y_naive2_df = y_naive2_df.rename(columns={'V1':'unique_id'})
    y_naive2_df = pd.wide_to_long(y_naive2_df, stubnames=["V"], i="unique_id", j="ds").reset_index()
    y_naive2_df = y_naive2_df.rename(columns={'V':'y_hat'})
    y_naive2_df = y_naive2_df.dropna()
    y_naive2_df['ds'] -= 1

    return y_naive2_df

###############################################################################
##### UTILS DATETIME ##########################################################
###############################################################################

def custom_offset(freq, x, int_ds=False):
    allowed_freqs= ('Y', 'M', 'W', 'H', 'Q', 'D')
    if freq not in allowed_freqs:
        raise ValueError(f'kind must be one of {allowed_kinds}')

    if int_ds:
        return x

    if freq == 'Y':
        return DateOffset(years = x)
    elif freq == 'M':
        return DateOffset(months = x)
    elif freq == 'W':
        return DateOffset(weeks = x)
    elif freq == 'H':
        return DateOffset(hours = x)
    elif freq == 'Q':
        return DateOffset(months = 3*x)
    elif freq == 'D':
        return DateOffset(days = x)

def fix_date(col, freq):
    if freq=='W':
      return date_to_start_week(col)
    if freq=='M':
      return date_to_start_month(col)
    if freq=='Q':
      return date_to_start_quarter(col)
    if freq=='Y':
      return date_to_start_year(col)
    else:
      return col

def date_to_start_week(col, week_starts_in=0):
    col = col - col.dt.weekday.apply(lambda x: timedelta(days=(x + week_starts_in + 1) % 7))
    return col

def date_to_start_month(col):
    col = col.apply(lambda x: x.replace(day=1))
    return col

def date_to_start_quarter(col):
    col = col - col.dt.month.apply(lambda x: custom_offset('M', (x-1) % 3))
    col = col.apply(lambda x: x.replace(day=1))
    return col

def date_to_start_year(col):
    return col.dt.year

def add_date(df, freq, data_directory=None, y_train_df=None):

    assert (data_directory is not None) or (y_train_df is not None)

    int_ds = freq=='Y'

    if y_train_df is None:
        m4_info = pd.read_csv(data_directory+'/M4-info.csv', usecols=['M4id','category', 'StartingDate'])
        m4_info = m4_info[m4_info['M4id'].str.startswith(freq)].reset_index(drop=True)
        m4_info['StartingDate'] = pd.to_datetime(m4_info['StartingDate'], dayfirst = True)

        # Some starting dates are parsed wrongly: ex 01-01-67 12:00	is parsed to 2067-01-01 12:00:00
        repair_dates = m4_info['StartingDate'].dt.strftime('%y').apply(lambda x: (int(x)<70) & (int(x)>17))
        m4_info.loc[repair_dates, 'StartingDate'] = m4_info.loc[repair_dates, 'StartingDate'].apply(lambda x: '19'+x.strftime('%y')+'-'+ x.strftime('%m-%d %H:%M:%S'))
        m4_info['StartingDate'] = pd.to_datetime(m4_info['StartingDate'])
        m4_info['StartingDate'] = fix_date(m4_info['StartingDate'], freq)
        m4_info = m4_info.rename(columns={'category': 'x'})

        dataset = df.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
    else:
        max_dates_train = y_train_df.groupby('unique_id').max()['ds'].reset_index()
        max_dates_train = max_dates_train.rename(columns={'ds': 'StartingDate'})
        dataset = df.merge(max_dates_train, how='left', on='unique_id')

    dataset.loc[:, 'ds'] = dataset['StartingDate']  + dataset['ds'].apply(lambda x: custom_offset(freq, x, int_ds))


    return dataset
