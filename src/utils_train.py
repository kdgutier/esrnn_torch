import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time

# from src.utils_data import dy_arrInput

from src.utils_config import ModelConfig
from src.utils_models import ES

def gaussian_noise(input_data, std=0.2):
  #shape = input_data.shape
  #noise = torch.autograd.Variable(torch.randn(shape).cuda() * std)
  size = input_data.size()
  noise = torch.autograd.Variable(input_data.data.new(size).normal_(0, std))
  return input_data + noise

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
