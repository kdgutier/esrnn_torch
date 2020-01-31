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
  X_df_train, y_df_train = M4_parser(dataset_name='Quarterly', mode='train')
  X_df_test, y_df_test = M4_parser(dataset_name='Quarterly', mode='test')

  esrnn = ESRNN(max_epochs=2, batch_size=4)
  esrnn.fit(X_df_train, y_df_train)

  y_df_hat = esrnn.predict(X_df_test)



if __name__ == '__main__':
  main()
