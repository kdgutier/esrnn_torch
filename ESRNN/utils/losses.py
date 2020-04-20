import torch
import torch.nn as nn

class PinballLoss(nn.Module):
  """ Pinball Loss
  Computes the pinball loss between y and y_hat.

  Parameters
  ----------  
  y: tensor
    actual values in torch tensor.
  y_hat: tensor (same shape as y)
    predicted values in torch tensor.
  tau: float, between 0 and 1
    the slope of the pinball loss, in the context of 
    quantile regression, the value of tau determines the
    conditional quantile level.
  
  Returns
  ----------
  pinball_loss:
    average accuracy for the predicted quantile
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
  """ Level Variability Loss
  Computes the variability penalty for the level.

  Parameters
  ----------
  levels: tensor with shape (batch, n_time)
    levels obtained from exponential smoothing component of ESRNN
  level_variability_penalty: float
    this parameter controls the strength of the penalization 
    to the wigglines of the level vector, induces smoothness
    in the output
  
  Returns
  ----------
  level_var_loss:
    wiggliness loss for the level vector
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

class StateLoss(nn.Module):
  pass

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


class DisaggregatedPinballLoss(nn.Module):
  """ Pinball Loss
  Computes the pinball loss between y and y_hat.

  Parameters
  ----------  
  y: tensor
    actual values in torch tensor.
  y_hat: tensor (same shape as y)
    predicted values in torch tensor.
  tau: float, between 0 and 1
    the slope of the pinball loss, in the context of 
    quantile regression, the value of tau determines the
    conditional quantile level.
  
  Returns
  ----------
  pinball_loss:
    average accuracy for the predicted quantile
  """
  def __init__(self, tau=0.5):
    super(DisaggregatedPinballLoss, self).__init__()
    self.tau = tau
  
  def forward(self, y, y_hat):
    delta_y = torch.sub(y, y_hat)
    pinball = torch.max(torch.mul(self.tau, delta_y), torch.mul((self.tau-1), delta_y))
    pinball = pinball.mean(axis=0).mean(axis=1)
    return pinball