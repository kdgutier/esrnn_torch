import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time

# from src.utils_data import dy_arrInput

from src.utils_config import ModelConfig
from src.utils_models import ES


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
    pinball = torch.max(torch.mul(self.tau, delta_y), torch.mul((self.tau-1), delta_y))
    pinball = pinball.mean()
    return pinball

class LevelVariabilityLoss(nn.Module):
  """Computes the variability penalty for the level.
  levels: levels obtained from exponential smoothing component of ESRNN.
          tensor with shape (batch, n_time).
  level_variability_penalty: float.
  return: level_var_loss
  """
  def __init__(self, level_variability_penalty):
    super(LevelVariabilityLoss, self).__init__()
    self.level_variability_penalty = level_variability_penalty

  def forward(self, levels):
    assert levels.shape[1] > 2
    level_prev = torch.log(levels[:, :-1])
    level_next = torch.log(levels[:, 1:])
    log_diff_of_levels = torch.sub(level_prev, level_next)

    log_diff_prev = log_diff_of_levels[:, :-1]
    log_diff_next = log_diff_of_levels[:, 1:]
    diff = torch.sub(log_diff_prev, log_diff_next)
    level_var_loss = diff**2
    level_var_loss = level_var_loss.mean() * self.level_variability_penalty
    return level_var_loss

class SmylLoss(nn.Module):
  """Computes the Smyl Loss that combines level variability with
  with Pinball loss.
  windows_y: tensor of actual values,
             shape (n_windows, batch_size, window_size).
  windows_y_hat: tensor of predicted values,
                 shape (n_windows, batch_size, window_size).
  levels: levels obtained from exponential smoothing component of ESRNN.
          tensor with shape (batch, n_time).
  return: smyl_loss.
  """
  def __init__(self, tau, level_variability_penalty=0.0):
    super(SmylLoss, self).__init__()
    self.pinball_loss = PinballLoss(tau)
    self.level_variability_loss = LevelVariabilityLoss(level_variability_penalty)

  def forward(self, windows_y, windows_y_hat, levels):
    smyl_loss = self.pinball_loss(windows_y, windows_y_hat)
    if self.level_variability_loss.level_variability_penalty>0:
      log_diff_of_levels = self.level_variability_loss(levels) 
      smyl_loss += log_diff_of_levels
    return smyl_loss


def train(mc, all_series):
  print(10*'='+' Training {} '.format(mc.dataset_name) + 10*'=')
  
  # Random Seeds
  torch.manual_seed(mc.copy)
  np.random.seed(mc.copy)

  # Model
  esrnn = ESRNN(mc)

  # Optimizers
  es_optimizer = optim.Adam(params=esrnn.es.parameters(),
                            lr=mc.learning_rate*mc.per_series_lr_multip, 
                            betas=(0.9, 0.999), eps=mc.gradient_eps)

  rnn_optimizer = optim.Adam(params=esrnn.rnn.parameters(),
                        lr=mc.learning_rate,
                        betas=(0.9, 0.999), eps=mc.gradient_eps)
  
  # Loss Functions
  smyl_loss = SmylLoss(tau=mc.tau,
                        level_variability_penalty=mc.level_variability_penalty)

  # training code
  for epoch in range(mc.max_epochs):
    start = time.time()
    
    losses = []
    for j in range(10):
      es_optimizer.zero_grad()
      rnn_optimizer.zero_grad()

      ts_object = all_series[j]
      windows_y, windows_y_hat, levels = esrnn(ts_object)
      
      loss = smyl_loss(windows_y, windows_y_hat, levels)
      losses.append(loss.data.numpy())
      loss.backward()
      torch.nn.utils.clip_grad_value_(esrnn.rnn.parameters(),
                                clip_value=mc.gradient_clipping_threshold)
      torch.nn.utils.clip_grad_value_(esrnn.es.parameters(),
                                clip_value=mc.gradient_clipping_threshold)
      rnn_optimizer.step()
      es_optimizer.step()
    
    if epoch >= (mc.max_epochs-mc.averaging_level):
      copy = epoch+mc.averaging_level-mc.max_epochs
      esrnn.save(copy=copy)

    print("========= Epoch {} finished =========".format(epoch))
    print("Training time: {}".format(time.time()-start))
    print("Forecast loss: {}".format(np.mean(losses)))

  print('Train finished!')
