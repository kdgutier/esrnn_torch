import numpy as np
import pandas as pd

import torch

class M4TS():
  def __init__(self, mc, category, y, id):
    self.id = id
    n = len(y)
    y = np.float32([y])
    y = torch.tensor(y).float()
    self.idxs = [id]
    if mc.lback>0:
      if (n - mc.lback * mc.output_size_i > 0):
        first = n - mc.lback * mc.output_size_i
        pastLast = n - (mc.lback-1) * mc.output_size_i
        self.y = y[:first]
        self.y_test = y[first:pastLast]
    else:
      self.y = y
    if (len(self.y) > mc.max_series_length):
      self.y = y[-mc.max_series_length:]

    category_dict = {'Demographic': 0, 'Finance': 1, 'Industry': 2,
                    'Macro': 3, 'Micro': 4, 'Other': 5}
    self.categories_vect = np.zeros((1,6))
    self.categories_vect[0,category_dict[category]] = 1
    self.categories_vect = torch.from_numpy(self.categories_vect).float()

def get_m4_all_series(mc, data='train'):
    m4_info = pd.read_csv(mc.data_dir+'M4-info.csv', usecols=['M4id','category'])

    if mc.dataset_name == 'm4hourly':
      dataset = pd.read_csv(mc.data_dir+data+'/Hourly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('H')].reset_index(drop=True)
    elif mc.dataset_name == 'm4daily':
      dataset = pd.read_csv(mc.data_dir+data+'/Daily-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('D')].reset_index(drop=True)
    elif mc.dataset_name == 'm4weekly':
      dataset = pd.read_csv(mc.data_dir+data+'/Weekly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('W')].reset_index(drop=True)
    elif mc.dataset_name == 'm4monthly':
      dataset = pd.read_csv(mc.data_dir+data+'/Monthly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('M')].reset_index(drop=True)
    elif mc.dataset_name == 'm4quarterly':
      dataset = pd.read_csv(mc.data_dir+data+'/Quarterly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('Q')].reset_index(drop=True)
    elif mc.dataset_name == 'm4yearly':
      dataset = pd.read_csv(mc.data_dir+data+'/Yearly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('Q')].reset_index(drop=True)

    all_series = []
    if mc.max_num_series==-1:
      max_num_series = len(dataset)
    else:
      max_num_series = min(len(dataset), mc.max_num_series)
    
    for i in range(max_num_series):
        row = dataset.loc[i]
        row = row[row.notnull()]
        category = m4_info.loc[i,'category']
        # omit row with column names
        y = row[1:].values
        #idx = row[0]
        m4_object = M4TS(mc, category, y, i)
        all_series.append(m4_object)
        
    return all_series
