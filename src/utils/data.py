import numpy as np
import torch

def long_to_wide(data):
    ts_map = {}
    for tmap, t in enumerate(data['ts'].unique()):
        ts_map[t] = tmap
    data['ts_map'] = data['ts'].map(ts_map)
    df_wide = data.pivot(index='unique_id', columns='ts_map')['y']
    
    x_unique = data[['unique_id', 'x']].groupby('unique_id').first()
    last_ts =  data[['unique_id', 'ts']].groupby('unique_id').last()
    assert len(x_unique)==len(data.unique_id.unique())
    df_wide['x'] = x_unique
    #df_wide['last_ts'] = last_ts
    df_wide = df_wide.reset_index().rename_axis(None, axis=1)
    
    ts_cols = data.ts_map.unique().tolist()
    X = df_wide.filter(items=['unique_id', 'x']).values
    y = df_wide.filter(items=ts_cols).values
    return X, y


class Batch():
    def __init__(self, mc, y, categories, idxs):
        self.id = id
        n = len(y)
        y = np.float32(y)        
        self.idxs = idxs
        self.y = y
        if (self.y.shape[1] > mc.max_series_length):
            self.y = y[:, -mc.max_series_length:]

        #self.last_ts = last_ts

        # Parse categoric data to 
        if mc.exogenous_size >0:
            self.categories = np.zeros((len(idxs), mc.exogenous_size))
            cols_idx = np.array([mc.category_to_idx[category] for category in categories])
            rows_idx = np.array(range(len(cols_idx)))
            self.categories[rows_idx, cols_idx] = 1
            self.categories = torch.from_numpy(self.categories).float()
        
        self.y = torch.tensor(y).float()


class Iterator(object):
    def __init__(self, mc, X, y, device=None,
                shuffle=False, random_seed=1):
        self.mc = mc
        self.X, self.y = X, y
        assert len(X)==len(y)
        
        self.batch_size = mc.batch_size
        self.shuffle = shuffle
        
        # Random Seeds
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.unique_idxs = np.unique(self.X[:, 0])
        assert len(self.unique_idxs)==len(self.X)
        self.n_series = len(self.unique_idxs)
        
        # Initialize batch iterator
        self.shuffle_dataset()
        self.b = 0
        self.n_batches = int(self.n_series / self.batch_size)

    def shuffle_dataset(self):
      """Return the examples in the dataset in order, or shuffled."""
      if self.shuffle:
        sort_key = np.random.choice(self.n_series, self.n_series, replace=False)
        self.X = self.X[sort_key]
        self.y = self.y[sort_key]
      else:
        sort_key = list(range(self.n_series))
      self.sort_key = {'unique_id': [self.unique_idxs[i] for i in sort_key],
                        'sort_key': sort_key}

    def get_trim_batch(self):
      # Compute the indexes of the minibatch.
      first = (self.b * self.batch_size)
      last = min((first + self.batch_size), self.n_series)

      # Extract values for batch
      batch_idxs = self.sort_key['sort_key'][first:last]

      batch_y = self.y[first:last]
      batch_categories = self.X[first:last, 1]

      last_numeric = np.count_nonzero(~np.isnan(batch_y), axis=1)
      min_len = min(last_numeric)

      # Trimming to match min_len
      #print("batch_y \n", batch_y)
      y_b = np.zeros((batch_y.shape[0], min_len))
      for i in range(batch_y.shape[0]):
          y_b[i] = batch_y[i,(last_numeric[i]-min_len):last_numeric[i]]
      batch_y = y_b
      #print("batch_y \n", batch_y)

      assert batch_y.shape[0] == len(batch_idxs) == len(batch_categories)
      assert batch_y.shape[1]>=1
      
      # Feed to Batch
      batch = Batch(mc=self.mc, y=batch_y, categories=batch_categories, idxs=batch_idxs)
      self.b = (self.b + 1) % self.n_batches
      return batch
    
    def get_batch(self):
      return self.get_trim_batch()

    def __len__(self):
      return self.n_batches
    
    def __iter__(self):
      pass
