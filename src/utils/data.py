import numpy as np
import torch

class tsObject():
    def __init__(self, mc, y, ts, categories, idxs):
        self.id = id
        n = len(y)
        y = np.float32([y])
        y = torch.tensor(y).float()
        self.idxs = idxs
        self.y = y
        if (len(self.y) > mc.max_series_length):
            self.y = y[-mc.max_series_length:]

        self.last_ts = ts[-1]

        # Parse categoric data to 
        if mc.exogenous_size >0:
            self.categories = np.zeros((len(idxs), mc.exogenous_size))
            cols_idx = np.array([mc.category_to_idx[category] for category in categories])
            rows_idx = np.array(range(len(cols_idx)))
            self.categories[rows_idx, cols_idx] = 1
            self.categories = torch.from_numpy(self.categories).float()
