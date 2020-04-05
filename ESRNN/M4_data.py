import os
from six.moves import urllib

import numpy as np
import pandas as pd


from ESRNN.utils_evaluation import Naive2
from ESRNN.utils_datetime import custom_offset

FREQ_DICT = {'Hourly': 'H',
             'Daily': 'D',
             'Weekly': 'W',
             'Monthly': 'M',
             'Quarterly': 'Q',
             'Yearly': 'Y'}

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

def M4_parser(dataset_name, directory, num_obs=1000000):
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
  frcy = FREQ_DICT[dataset_name]

  m4_info = pd.read_csv(data_directory+'/M4-info.csv', usecols=['M4id','category', 'StartingDate'])
  m4_info = m4_info[m4_info['M4id'].str.startswith(dataset_name[0])].reset_index(drop=True)
  m4_info['StartingDate'] = pd.to_datetime(m4_info['StartingDate'])

  # Some starting dates are parsed wrongly: ex 01-01-67 12:00	is parsed to 2067-01-01 12:00:00
  repair_dates = m4_info['StartingDate'].dt.strftime('%y').apply(lambda x: (int(x)<70) & (int(x)>17))
  m4_info.loc[repair_dates, 'StartingDate'] = m4_info.loc[repair_dates, 'StartingDate'].apply(lambda x: '19'+x.strftime('%y')+'-'+ x.strftime('%m-%d %H:%M:%S'))

  # train data
  train_path='{}{}-train.csv'.format(train_directory, dataset_name)
  dataset = pd.read_csv(train_path).head(num_obs)
  dataset = dataset.rename(columns={'V1':'unique_id'})

  dataset = pd.wide_to_long(dataset, stubnames=["V"], i="unique_id", j="ds").reset_index()
  dataset = dataset.rename(columns={'V':'y'})
  dataset = dataset.dropna()
  dataset = dataset.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  # Some yearly time series has more than 200 observations
  # Example: Y13190 has 835 obs
  # In this cases we only use the last 200 obs
  if frcy == 'Y':
      fix_ids = dataset.loc[dataset['ds'] >= 200, 'unique_id'].unique()
      if  not (fix_ids.size == 0):
          print('Some yearly time series have more than 200 observations')
          print('Returning only last 200 obs of this time series ({} time series)\n'.format(len(fix_ids)))
          #
          # fixing ds for problematic ts
          problematic_ts = dataset[dataset['unique_id'].isin(fix_ids)]
          problematic_ts = problematic_ts.groupby('unique_id').tail(200)
          min_ds = problematic_ts.groupby('unique_id').min().reset_index()[['unique_id', 'ds']]
          min_ds.rename(columns={'ds': 'min_ds'}, inplace=True)

          problematic_ts = problematic_ts.merge(min_ds, how='left', on=['unique_id'])
          problematic_ts['ds'] -= (problematic_ts['min_ds'] - 2)
          problematic_ts.drop(columns='min_ds', inplace=True)

          non_problematic_ts = dataset[~dataset['unique_id'].isin(fix_ids)]

          dataset = pd.concat([non_problematic_ts, problematic_ts])
          dataset = dataset.sort_values(['unique_id', 'ds'])

          del non_problematic_ts, problematic_ts

  dataset.loc[:, 'ds'] = pd.to_datetime(dataset['StartingDate']) + dataset['ds'].apply(lambda x: custom_offset(frcy, x-2))

  dataset.drop(columns=['M4id'], inplace=True)
  dataset = dataset.rename(columns={'category': 'x'})
  dataset.sort_values(by=['unique_id', 'ds'], inplace=True)
  X_train_df = dataset.filter(items=['unique_id', 'ds', 'x'])
  y_train_df = dataset.filter(items=['unique_id', 'ds', 'y'])

  max_dates = X_train_df.groupby('unique_id').agg({'ds': 'count'}).reset_index()

  # test data
  test_path='{}{}-test.csv'.format(test_directory, dataset_name)
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
  dataset.loc[:, 'ds'] = pd.to_datetime(dataset['StartingDate']) + dataset['ds'].apply(lambda x: custom_offset(frcy, x-2))

  X_test_df = dataset.filter(items=['unique_id', 'x', 'ds'])
  y_test_df = dataset.filter(items=['unique_id', 'y', 'ds'])

  return X_train_df, y_train_df, X_test_df, y_test_df

def naive2_predictions(dataset_name, directory, num_obs):
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
    """
    # Read train and test data
    _, y_train_df, _, y_test_df = M4_parser(dataset_name, directory, num_obs)

    seas_dict = {'Hourly': {'seasonality': 24, 'input_size': 24,
                           'output_size': 48},
                 'Daily': {'seasonality': 7, 'input_size': 7,
                           'output_size': 14},
                 'Weekly': {'seasonality': 52, 'input_size': 52,
                            'output_size': 13},
                 'Monthly': {'seasonality': 12, 'input_size': 12,
                             'output_size':24},
                 'Quarterly': {'seasonality': 4, 'input_size': 4,
                               'output_size': 8},
                 'Yearly': {'seasonality': 1, 'input_size': 4,
                            'output_size': 6}}

    seasonality = seas_dict[dataset_name]['seasonality']
    input_size = seas_dict[dataset_name]['input_size']
    output_size = seas_dict[dataset_name]['output_size']
    frcy = FREQ_DICT[dataset_name]


    print('Preparing {} dataset'.format(dataset_name))
    print('Preparing Naive2 {} dataset predictions'.format(dataset_name))

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
        y_naive2['ds'] = np.arange(1, output_size+1)#pd.date_range(start=y_id.ds.max(),
                         #               periods=output_size+1, freq=frcy)[1:]
        y_naive2['StartingDate'] = y_id.ds.max()
        y_naive2['ds'] = pd.to_datetime(y_naive2['StartingDate']) + y_naive2['ds'].apply(lambda x: custom_offset(frcy, x))
        y_naive2.drop(columns='StartingDate', inplace=True)

        y_naive2['unique_id'] = unique_id
        y_naive2['y_hat'] = Naive2(seasonality).fit(y_id.y.to_numpy()).predict(output_size)
        y_naive2_df = y_naive2_df.append(y_naive2)

    y_naive2_df = y_test_df.merge(y_naive2_df, on=['unique_id', 'ds'], how='left')
    y_naive2_df.rename(columns={'y_hat': 'y_hat_naive2'}, inplace=True)

    results_dir = directory + '/results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    naive2_file = results_dir + '/{}-naive2predictions_{}.csv'.format(dataset_name, num_obs)
    y_naive2_df.to_csv(naive2_file, encoding='utf-8', index=None)
    return y_naive2_df

def prepare_M4_data(dataset_name, directory, num_obs):
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

  X_train_df, y_train_df, X_test_df, y_test_df = M4_parser(dataset_name, directory, num_obs)

  results_dir = directory + '/results'
  if not os.path.exists(results_dir):
      os.mkdir(results_dir)

  naive2_file = results_dir + '/{}-naive2predictions_{}.csv'.format(dataset_name, num_obs)
  if not os.path.exists(naive2_file):
    y_naive2_df = naive2_predictions(dataset_name, directory, num_obs)

  else:
    y_naive2_df = pd.read_csv(naive2_file)
    y_naive2_df['ds'] = pd.to_datetime(y_naive2_df['ds'])

  return X_train_df, y_train_df, X_test_df, y_naive2_df
