import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pathlib import Path
from ESRNN.utils.config import ModelConfig
from ESRNN.utils.ESRNN import _ESHolt
from ESRNN.utils.losses import SmylLoss
from ESRNN.utils.data import Iterator


class ESHolt(object):
  def __init__(self, max_epochs=15, batch_size=1, learning_rate=1e-3, gradient_eps=1e-6, gradient_clipping_threshold=20,
                lr_scheduler_step_size=9, noise_std=0.001, level_variability_penalty=80, tau=0.5,
                seasonality=4, input_size=4, output_size=8, frequency='D', max_periods=20, device='cpu', root_dir='./'):
    super(ESHolt, self).__init__()
    self.mc = ModelConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=learning_rate, per_series_lr_multip=np.nan, 
                          gradient_eps=gradient_eps, gradient_clipping_threshold=gradient_clipping_threshold, lr_scheduler_step_size=lr_scheduler_step_size, 
                          noise_std=noise_std, level_variability_penalty=level_variability_penalty, tau=tau,
                          state_hsize=np.nan, dilations=[], add_nl_layer=False, 
                          seasonality=seasonality, input_size=input_size, output_size=output_size,
                          frequency=frequency, max_periods=max_periods, device=device, root_dir=root_dir)
  
  def train(self, dataloader, random_seed):
    print(10*'='+' Training ESHolt ' + 10*'=' + '\n')

    # Optimizers
    optimizer = optim.Adam(params=self.esholt.parameters(),
                              lr=self.mc.learning_rate, 
                              betas=(0.9, 0.999), eps=self.mc.gradient_eps)
    
    scheduler = StepLR(optimizer=optimizer,
                       step_size=self.mc.lr_scheduler_step_size,
                       gamma=0.9)

    # Loss Functions
    smyl_loss = SmylLoss(tau=self.mc.tau, level_variability_penalty=self.mc.level_variability_penalty)

    # training code
    for epoch in range(self.mc.max_epochs):
      start = time.time()
      if self.shuffle:
        dataloader.shuffle_dataset(random_seed=epoch)
      losses = []
      for j in range(dataloader.n_batches):
        optimizer.zero_grad()

        batch = dataloader.get_batch()
        windows_y, windows_y_hat, levels = self.esholt(batch)
        
        loss = smyl_loss(windows_y, windows_y_hat, levels)
        losses.append(loss.data.numpy())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.esholt.parameters(), self.mc.gradient_clipping_threshold)
        optimizer.step()

      # Decay learning rate
      scheduler.step()

      print("========= Epoch {} finished =========".format(epoch))
      print("Training time: {}".format(time.time()-start))
      print("Forecast loss: {}".format(np.mean(losses)))

    print('Train finished!')