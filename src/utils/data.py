import numpy as np
import pandas as pd

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
    df_wide['last_ts'] = last_ts
    df_wide = df_wide.reset_index().rename_axis(None, axis=1)
    
    ts_cols = data.ts_map.unique().tolist()
    y = df_wide.filter(items=['unique_id'] + ts_cols)
    X = df_wide.filter(items=['unique_id', 'x', 'last_ts'])
    return X, y


class Batch():
    def __init__(self, mc, y, last_ts, categories, idxs):
        self.id = id
        n = len(y)
        y = np.float32(y)        
        self.idxs = idxs
        self.y = y
        if (self.y.shape[1] > mc.max_series_length):
            self.y = y[:, -mc.max_series_length:]

        self.last_ts = last_ts

        # Parse categoric data to 
        if mc.exogenous_size >0:
            self.categories = np.zeros((len(idxs), mc.exogenous_size))
            cols_idx = np.array([mc.category_to_idx[category] for category in categories])
            rows_idx = np.array(range(len(cols_idx)))
            self.categories[rows_idx, cols_idx] = 1
            self.categories = torch.from_numpy(self.categories).float()
        
        y = torch.tensor(y).float()


class Iterator(object):
    def __init__(self, mc, X, y, device=None,
                shuffle=False, random_seed=1):
        self.mc = mc
        self.X, self.y = X, y
        assert len(X)==len(y)
        
        self.batch_size = mc.batch_size
        self.shuffle = shuffle
        
        self.random_seed = random_seed
        self.unique_idxs = self.X['unique_id'].unique()
        self.n_series = len(self.unique_idxs)
        
        self.shuffle_dataset()

    def shuffle_dataset(self):
        """Return the examples in the dataset in order, or shuffled."""
        if self.shuffle:
            print("self.shuffle", self.shuffle)
            sort_key = np.random.choice(self.n_series, self.n_series, replace=False)
            sort_key = {'unique_id': [self.unique_idxs[i] for i in sort_key],
                        'sort_key': list(range(self.n_series))}
            self.sort_key = pd.DataFrame.from_dict(sort_key)
            self.X = self.sort_key.merge(self.X, on=['unique_id'])
            self.y = self.sort_key.merge(self.y, on=['unique_id'])
        else:
            print("self.shuffle", self.shuffle)
            sort_key = {'unique_id': self.unique_idxs,
                        'sort_key': list(range(self.n_series))}
            self.sort_key = pd.DataFrame.from_dict(sort_key)
            self.X = self.sort_key.merge(self.X, on=['unique_id'])
            self.y = self.sort_key.merge(self.y, on=['unique_id'])
    
    def panel_to_batches(self):
        """
        Receives panel and creates batches list wit ts objects.
        Parameters:
            X: SORTED array-like or sparse matrix, shape (n_samples, n_features)
                Test or validation data for panel, with column 'unique_id', date stamp 'ds' and 'y'.
        Returns:
            tsobject_list : list of ts objects
        """
        n_batches = int(self.n_series / self.batch_size)

        batches = []
        for b in range(n_batches):
            # Compute the offset of the minibatch.
            offset = (b * self.batch_size) % self.n_series
            
            # Extract values for batch
            batch_idxs = self.sort_key.iloc[offset:(offset + self.batch_size)]['sort_key'].values
            batch_y = self.y.iloc[offset:(offset + self.batch_size), 2:].values

            batch_categories = self.X.iloc[offset:(offset + self.batch_size)]['x'].values
            batch_last_ts = self.X.iloc[offset:(offset + self.batch_size)]['last_ts'].values

            assert batch_y.shape[0] == len(batch_idxs) == len(batch_last_ts) == len(batch_categories)
            assert batch_y.shape[1]>2
            
            # Feed to Batch
            batch = Batch(mc=self.mc, y=batch_y, last_ts=batch_last_ts, 
                              categories=batch_categories, idxs=batch_idxs)

            batches.append(batch)
        return batches
