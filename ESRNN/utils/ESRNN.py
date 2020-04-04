import torch
import torch.nn as nn
from ESRNN.utils.DRNN import DRNN
import numpy as np

#import torch.jit as jit


class _ES(nn.Module):
  def __init__(self, mc):
    super(_ES, self).__init__()
    self.mc = mc
    self.n_series = self.mc.n_series
    self.output_size = self.mc.output_size
    assert len(self.mc.seasonality) in [0, 1, 2]

  def gaussian_noise(self, input_data, std=0.2):
    size = input_data.size()
    noise = torch.autograd.Variable(input_data.data.new(size).normal_(0, std))
    return input_data + noise

  #@jit.script_method
  def compute_levels_seasons(self, y, idxs):
    pass

  def normalize(self, y, level, seasonalities):
    pass

  def predict(self, trend, levels, seasonalities):
    pass

  def forward(self, ts_object):
    # parse mc
    input_size = self.mc.input_size
    output_size = self.mc.output_size
    exogenous_size = self.mc.exogenous_size
    noise_std = self.mc.noise_std
    seasonality = self.mc.seasonality
    batch_size = len(ts_object.idxs)

    # Parse ts_object
    y = ts_object.y
    idxs = ts_object.idxs
    n_series, n_time = y.shape
    if self.training:
      windows_end = n_time-input_size-output_size+1
      windows_range = range(windows_end)
    else:
      windows_start = n_time-input_size-output_size+1
      windows_end = n_time-input_size+1

      windows_range = range(windows_start, windows_end)
    n_windows = len(windows_range)
    assert n_windows>0

    # Initialize windows, levels and seasonalities
    levels, seasonalities = self.compute_levels_seasons(y, idxs)
    windows_y_hat = torch.zeros((n_windows, batch_size, input_size+exogenous_size),
                                device=self.mc.device)
    windows_y = torch.zeros((n_windows, batch_size, output_size),
                            device=self.mc.device)

    for i, window in enumerate(windows_range):
      # Windows yhat
      y_hat_start = window
      y_hat_end = input_size + window

      # Y_hat deseasonalization and normalization
      window_y_hat = self.normalize(y=y[:, y_hat_start:y_hat_end],
                                    level=levels[:, [y_hat_end-1]],
                                    seasonalities=seasonalities,
                                    start=y_hat_start, end=y_hat_end)

      if self.training:
        window_y_hat = self.gaussian_noise(window_y_hat, std=noise_std)

      # Concatenate categories
      if exogenous_size>0:
        window_y_hat = torch.cat((window_y_hat, ts_object.categories), 1)

      windows_y_hat[i, :, :] += window_y_hat

      # Windows y (for loss during train)
      if self.training:
        y_start = y_hat_end
        y_end = y_start+output_size
        # Y deseasonalization and normalization
        window_y = self.normalize(y=y[:, y_start:y_end],
                                  level=levels[:, [y_start]],
                                  seasonalities=seasonalities,
                                  start=y_start, end=y_end)
        windows_y[i, :, :] += window_y

    return windows_y_hat, windows_y, levels, seasonalities

class _ESM(_ES):
  def __init__(self, mc):
    super(_ESM, self).__init__(mc)
    # Level and Seasonality Smoothing parameters
    # 1 level, S seasonalities, S init_seas
    embeds_size = 1 + len(self.mc.seasonality) + sum(self.mc.seasonality)
    init_embeds = torch.ones((self.n_series, embeds_size)) * 0.5
    self.embeds = nn.Embedding(self.n_series, embeds_size)
    self.embeds.weight.data.copy_(init_embeds)
    self.register_buffer('seasonality', torch.LongTensor(self.mc.seasonality))

  #@jit.script_method
  def compute_levels_seasons(self, y, idxs):
    """
    Computes levels and seasons
    """
    # Lookup parameters per serie
    #seasonality = self.seasonality
    embeds = self.embeds(idxs)
    lev_sms = torch.sigmoid(embeds[:, 0])

    # Initialize seasonalities
    seas_prod = torch.ones(len(y[:,0])).to(y.device)
    #seasonalities1 = torch.jit.annotate(List[Tensor], [])
    #seasonalities2 = torch.jit.annotate(List[Tensor], [])
    seasonalities1 = []
    seasonalities2 = []
    seas_sms1 = torch.ones(1).to(y.device)
    seas_sms2 = torch.ones(1).to(y.device)

    if len(self.seasonality)>0:
      seas_sms1 = torch.sigmoid(embeds[:, 1])
      init_seas1 = torch.exp(embeds[:, 2:(2+self.seasonality[0])]).unbind(1)
      assert len(init_seas1) == self.seasonality[0]

      for i in range(len(init_seas1)):
        seasonalities1 += [init_seas1[i]]
      seasonalities1 += [init_seas1[0]]
      seas_prod = seas_prod * init_seas1[0]

    if len(self.seasonality)==2:
      seas_sms2 = torch.sigmoid(embeds[:, 2+self.seasonality[0]])
      init_seas2 = torch.exp(embeds[:, 3+self.seasonality[0]:]).unbind(1)
      assert len(init_seas2) == self.seasonality[1]

      for i in range(len(init_seas2)):
        seasonalities2 += [init_seas2[i]]
      seasonalities2 += [init_seas2[0]]
      seas_prod = seas_prod * init_seas2[0]

    # Initialize levels
    #levels = torch.jit.annotate(List[Tensor], [])
    levels = []
    levels += [y[:,0]/seas_prod]

    # Recursive seasonalities and levels
    ys = y.unbind(1)
    n_time = len(ys)
    for t in range(1, n_time):

      seas_prod_t = torch.ones(len(y[:,t])).to(y.device)
      if len(self.seasonality)>0:
        seas_prod_t = seas_prod_t * seasonalities1[t]
      if len(self.seasonality)==2:
        seas_prod_t = seas_prod_t * seasonalities2[t]

      newlev = lev_sms * (ys[t] / seas_prod_t) + (1-lev_sms) * levels[t-1]
      levels += [newlev]

      if len(self.seasonality)==1:
        newseason1 = seas_sms1 * (ys[t] / newlev) + (1-seas_sms1) * seasonalities1[t]
        seasonalities1 += [newseason1]

      if len(self.seasonality)==2:
        newseason1 = seas_sms1 * (ys[t] / (newlev * seasonalities2[t])) + \
                     (1-seas_sms1) * seasonalities1[t]
        seasonalities1 += [newseason1]
        newseason2 = seas_sms2 * (ys[t] / (newlev * seasonalities1[t])) + \
                     (1-seas_sms2) * seasonalities2[t]
        seasonalities2 += [newseason2]

    levels = torch.stack(levels).transpose(1,0)

    #seasonalities = torch.jit.annotate(List[Tensor], [])
    seasonalities = []

    if len(self.seasonality)>0:
      seasonalities += [torch.stack(seasonalities1).transpose(1,0)]

    if len(self.seasonality)==2:
      seasonalities += [torch.stack(seasonalities2).transpose(1,0)]

    return levels, seasonalities

  def normalize(self, y, level, seasonalities, start, end):
    # Deseasonalization and normalization
    y_n = y / level
    for s in range(len(self.seasonality)):
      y_n /= seasonalities[s][:, start:end]
    y_n = torch.log(y_n)
    return y_n

  def predict(self, trend, levels, seasonalities):
    output_size = self.mc.output_size
    seasonality = self.mc.seasonality
    n_time = levels.shape[1]

    # Denormalize
    trend = torch.exp(trend)

    # Completion of seasonalities if prediction horizon is larger than seasonality
    # Naive2 like prediction, to avoid recursive forecasting
    for s in range(len(seasonality)):
      if output_size > seasonality[s]:
        repetitions = int(np.ceil(output_size/seasonality[s]))-1
        last_season = seasonalities[s][:, -seasonality[s]:]
        extra_seasonality = last_season.repeat((1, repetitions))
        seasonalities[s] = torch.cat((seasonalities[s], extra_seasonality), 1)

    # Deseasonalization and normalization (inverse)
    y_hat = trend * levels[:,[n_time-1]]
    for s in range(len(seasonality)):
      y_hat *= seasonalities[s][:, n_time:(n_time+output_size)]

    return y_hat

class _RNN(nn.Module):
  def __init__(self, mc):
    super(_RNN, self).__init__()
    self.mc = mc
    self.layers = len(mc.dilations)

    layers = []
    for grp_num in range(len(mc.dilations)):
      if grp_num == 0:
        input_size = mc.input_size + mc.exogenous_size
      else:
        input_size = mc.state_hsize
      layer = DRNN(input_size,
                   mc.state_hsize,
                   n_layers=len(mc.dilations[grp_num]),
                   dilations=mc.dilations[grp_num],
                   cell_type=mc.cell_type)
      layers.append(layer)

    self.rnn_stack = nn.Sequential(*layers)

    if self.mc.add_nl_layer:
      self.MLPW  = nn.Linear(mc.state_hsize, mc.state_hsize)

    self.adapterW  = nn.Linear(mc.state_hsize, mc.output_size)

  def forward(self, input_data):
    for layer_num in range(len(self.rnn_stack)):
      residual = input_data
      output, _ = self.rnn_stack[layer_num](input_data)
      if layer_num > 0:
        output += residual
      input_data = output

    if self.mc.add_nl_layer:
      input_data = self.MLPW(input_data)
      input_data = torch.tanh(input_data)

    input_data = self.adapterW(input_data)
    return input_data


class _ESRNN(nn.Module):
  def __init__(self, mc):
    super(_ESRNN, self).__init__()
    self.mc = mc
    self.es = _ESM(mc).to(self.mc.device)
    self.rnn = _RNN(mc).to(self.mc.device)

  def forward(self, ts_object):
    # ES Forward
    windows_y_hat, windows_y, levels, seasonalities = self.es(ts_object)

    # RNN Forward
    windows_y_hat = self.rnn(windows_y_hat)

    return windows_y, windows_y_hat, levels

  def predict(self, ts_object):
    # ES Forward
    windows_y_hat, _, levels, seasonalities = self.es(ts_object)

    # RNN Forward
    windows_y_hat = self.rnn(windows_y_hat)
    trend = windows_y_hat[-1,:,:] # Last observation prediction

    y_hat = self.es.predict(trend, levels, seasonalities)
    return y_hat
