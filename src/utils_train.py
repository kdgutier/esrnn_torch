import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import time

# from src.utils_data import dy_arrInput

from src.utils_config import ModelConfig
from src.utils_models import ESRNN

class PinballLoss(nn.Module):
    """Computes the pinball loss between y and y_hat.
    y: actual values in torch tensor.
    y_hat: predicted values in torch tensor.
    tau: a float between 0 and 1 the slope of the pinball loss. In the context
    of quantile regression, the value of alpha determine the conditional
    quantile level.
    return: pinball_loss
    """
  def __init__(self, tau=0.5):
    super(PinballLoss, self).__init__()
    self.tau = tau
  
  def forward(self, y, y_hat):
    delta_y = torch.sub(y, y_hat)
    pinball = torch.max(torch.mul(tau, delta_y), torch.mul((tau-1), delta_y))
    pinball = pinball.mean()
    return pinball

class LevelVAriabilityLoss(nn.Module):
    """Computes the variability penalty for the level.
    y: level values in torch tensor.
    level_variability_penalty: 
    return: level_var_loss
    """
    def __init__(self, level_variability_penalty):
      super(LevelVAriabilityLoss, self).__init__()
      self.level_variability_penalty = level_variability_penalty

    def forward(self, y):
      y_prev = y[:-1]
      y_next = y[1:]
      diff =  torch.sub(y_prev, y_next)
      level_var_loss = diff**2
      level_var_loss = level_var_loss.mean() * self.level_variability_penalty
    return level_var_loss


def train(mc, all_series):
    print(10*'='+' Training {}'.format(mc.dataset_name) + 10*'=')

    # Model
    esrnn = ESRNN(mc)

    # Trainers
    per_series_trainer = dy.AdamTrainer(esrnn.es.pc, alpha=mc.learning_rate*mc.per_series_lr_multip, 
                                    beta_1=0.9, beta_2=0.999, eps=mc.gradient_eps)
    per_series_trainer.set_clip_threshold(mc.gradient_clipping_threshold)

    trainer = dy.AdamTrainer(esrnn.rnn.pc, alpha=mc.learning_rate,
                            beta_1=0.9, beta_2=0.999, eps=mc.gradient_eps)
    trainer.set_clip_threshold(mc.gradient_clipping_threshold)

    # training code
    for epoch in range(mc.max_epochs):
        start = time.time()
        forecast_losses = []
        lev_variability_losses = []
        state_losses = []
        
        for j in range(len(all_series)):
            dy.renew_cg() # new computation graph
            ts_object = all_series[j]
            
            esrnn.declare_expr(ts_object)
            levels_ex, seasonalities_ex, log_diff_of_levels = esrnn.compute_levels_seasons(ts_object)
            
            losses = []
            for i in range(mc.input_size-1, len(ts_object.y)-mc.output_size):
                input_start = i+1-mc.input_size
                input_end = i+1
                
                # Deseasonalization and normalization
                input_ex = ts_object.y[input_start:input_end]
                input_ex = dy_arrInput(input_ex)
                season_ex = seasonalities_ex[input_start:input_end]
                season_ex = dy.concatenate(season_ex)
                input_ex = dy.cdiv(input_ex, season_ex)
                input_ex = dy.cdiv(input_ex, levels_ex[i])
                input_ex = dy.noise(dy.log(input_ex), mc.noise_std)
                
                # Concatenate categories
                categories_ex = dy_arrInput(ts_object.categories_vect)
                input_ex = dy.concatenate([input_ex, categories_ex])
        
                output_ex = esrnn.rnn(input_ex)
                
                labels_start = i+1
                labels_end = i+1+mc.output_size
                
                # Deseasonalization and normalization
                labels_ex = ts_object.y[labels_start:labels_end]
                labels_ex = dy_arrInput(labels_ex)
                season_ex = seasonalities_ex[labels_start:labels_end]
                season_ex = dy.concatenate(season_ex)
                labels_ex = dy.cdiv(labels_ex, season_ex)
                labels_ex = dy.cdiv(dy.log(labels_ex), levels_ex[i])
                
                loss_ex = pinball_loss(labels_ex, output_ex)
                losses.append(loss_ex)
            
            # Losses
            forecloss_ex = dy.average(losses)
            loss_ex = forecloss_ex
            forecast_losses.append(forecloss_ex.npvalue())

            if mc.level_variability_penalty>0:
                level_var_loss_ex = level_variability_loss(log_diff_of_levels, 
                                                        mc.level_variability_penalty)
                loss_ex += level_var_loss_ex
                lev_variability_losses.append(level_var_loss_ex.npvalue())
            
            loss_ex.backward()
            trainer.update()
            per_series_trainer.update()
        
        if epoch >= (mc.max_epochs-mc.averaging_level):
            copy = epoch+mc.averaging_level-mc.max_epochs
            esrnn.save(copy=copy)

        print("========= Epoch {} finished =========".format(epoch))
        print("Training time: {}".format(time.time()-start))
        print("Forecast loss: {}".format(np.mean(forecloss_ex.npvalue())))

    print('Train finished!')
