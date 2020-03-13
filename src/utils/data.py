import numpy as np
import torch


class Batch():
  def __init__(self, mc, y, last_ds, categories, idxs):
    # Parse Model config
    exogenous_size = mc.exogenous_size
    device = mc.device

    # y: time series values
    n = len(y)
    y = np.float32(y)
    self.idxs = torch.LongTensor(idxs).to(device)
    self.y = y
    if (self.y.shape[1] > mc.max_series_length):
        y = y[:, -mc.max_series_length:]
    self.y = torch.tensor(y).float()

    # last_ds: last time for prediction purposes
    self.last_ds = last_ds

    # categories: exogenous categoric data
    if exogenous_size >0:
      self.categories = np.zeros((len(idxs), exogenous_size))
      cols_idx = np.array([mc.category_to_idx[category] for category in categories])
      rows_idx = np.array(range(len(cols_idx)))
      self.categories[rows_idx, cols_idx] = 1
      self.categories = torch.from_numpy(self.categories).float()

    self.y = self.y.to(device)
    self.categories = self.categories.to(device)


class Iterator(object):
  """ Time Series Iterator.

  Parameters
  ----------
  mc: ModelConfig object
    ModelConfig object with inherited hyperparameters:
    batch_size, and exogenous_size, from the ESRNN 
    initialization.
  X: array, shape (n_unique_id, 3)
    Panel array with unique_id, last date stamp and 
    exogenous variable.
  y: array, shape (n_unique_id, n_time)
    Panel array in wide format with unique_id, last
    date stamp and time series values.
  Returns
  ----------
  self : object
    Iterator method get_batch() returns a batch of time
    series objects defined by the Batch class.
  """
  def __init__(self, mc, X, y, weights=None):
    if weights is not None:
      assert len(weights)==len(X)
      train_ids = np.where(weights==1)[0]
      self.X = X[train_ids,:]
      self.y = y[train_ids,:]
    else:
      self.X = X
      self.y = y
    assert len(X)==len(y)
    
    # Parse Model config
    self.mc = mc
    self.batch_size = mc.batch_size

    self.unique_idxs = np.unique(self.X[:, 0])
    assert len(self.unique_idxs)==len(self.X)
    self.n_series = len(self.unique_idxs)

    #assert self.batch_size <= self.n_series 
    
    # Initialize batch iterator
    self.b = 0
    self.n_batches = int(np.ceil(self.n_series / self.batch_size))
    shuffle = list(range(self.n_series))
    self.sort_key = {'unique_id': [self.unique_idxs[i] for i in shuffle],
                     'sort_key': shuffle}
  
  def update_batch_size(self, new_batch_size):
    self.batch_size = new_batch_size
    assert self.batch_size <= self.n_series
    self.n_batches = int(np.ceil(self.n_series / self.batch_size))

  def shuffle_dataset(self, random_seed=1):
    """Return the examples in the dataset in order, or shuffled."""
    # Random Seed
    np.random.seed(random_seed)
    self.random_seed = random_seed
    shuffle = np.random.choice(self.n_series, self.n_series, replace=False)
    self.X = self.X[shuffle]
    self.y = self.y[shuffle]

    old_sort_key = self.sort_key['sort_key']
    old_unique_idxs = self.sort_key['unique_id']
    self.sort_key = {'unique_id': [old_unique_idxs[i] for i in shuffle],
                     'sort_key': [old_sort_key[i] for i in shuffle]}

  def get_trim_batch(self, unique_id):
    if unique_id==None:
      # Compute the indexes of the minibatch.
      first = (self.b * self.batch_size)
      last = min((first + self.batch_size), self.n_series)
    else:
      # Obtain unique_id index
      assert unique_id in self.sort_key['unique_id'], "unique_id, not fitted"
      first = self.sort_key['unique_id'].index(unique_id)
      last = first+1

    # Extract values for batch
    unique_idxs = self.sort_key['unique_id'][first:last]
    batch_idxs = self.sort_key['sort_key'][first:last]

    batch_y = self.y[first:last]
    batch_categories = self.X[first:last, 1]
    batch_last_ds = self.X[first:last, 2]

    len_series = np.count_nonzero(~np.isnan(batch_y), axis=1)
    min_len = min(len_series)
    last_numeric = (~np.isnan(batch_y)).cumsum(1).argmax(1)+1

    # Trimming to match min_len
    y_b = np.zeros((batch_y.shape[0], min_len))
    for i in range(batch_y.shape[0]):
      y_b[i] = batch_y[i,(last_numeric[i]-min_len):last_numeric[i]]
    batch_y = y_b

    assert not np.isnan(batch_y).any(), \
           "clean np.nan's from unique_idxs: {}".format(unique_idxs)
    assert batch_y.shape[0]==len(batch_idxs)==len(batch_last_ds)==len(batch_categories)
    assert batch_y.shape[1]>=1

    # Feed to Batch
    batch = Batch(mc=self.mc, y=batch_y, last_ds=batch_last_ds,
                  categories=batch_categories, idxs=batch_idxs)
    self.b = (self.b + 1) % self.n_batches
    return batch
    
  def get_batch(self, unique_id=None):
    return self.get_trim_batch(unique_id)

  def __len__(self):
    return self.n_batches
    
  def __iter__(self):
    pass
