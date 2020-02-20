# lovingly borrowed from https://github.com/zalandoresearch/pytorch-dilated-rnn
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
# https://discuss.pytorch.org/t/access-gates-of-lstm-gru/12399/3
# https://discuss.pytorch.org/t/getting-modification-to-lstmcell-to-pass-over-to-cuda/16748
# https://medium.com/@andre.holzner/lstm-cells-in-pytorch-fab924a78b1c
# https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-in-pytorch-lstms-in-depth-part-1/

import torch
import torch.nn as nn
import torch.autograd as autograd

use_cuda = torch.cuda.is_available()

import warnings
from torch.autograd import NestedIOFunction
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
#from .thnn import rnnFusedPointwise as fusedBackend

try:
    import torch.backends.cudnn.rnn
except ImportError:
    pass


class ResidualLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, dropout=0.):
    super(ResidualLSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.w_ih = nn.Parameter(torch.Tensor(hidden_size * 4, input_size))
    self.w_hh = nn.Parameter(torch.Tensor(hidden_size * 4, hidden_size))
    self.b_ih = nn.Parameter(torch.Tensor(hidden_size * 4))
    self.b_hh = nn.Parameter(torch.Tensor(hidden_size * 4))
  
  def init_weights(self):
    for p in self.parameters():
      if p.data.ndimension() >= 2:
        nn.init.xavier_uniform_(p.data)
      else:
        nn.init.zeros_(p.data)

  def ResidualLSTMCell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0) #compatibility hack
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate     = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate   = torch.tanh(cellgate)
    outgate    = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy
  
  def forward(self, x, hidden=None):
    """
    Like LSTM inputs is of shape (seq_len, batch_size, input_size)
    Like LSTM inputs (num_layers \* num_directions, batch, hidden_size)
    num_layers is for compatibility hack
    """
    seq_size, batch_size, input_size = x.size()
    output = []
    if hidden is None:
      hx, cx = (torch.zeros(batch_size, self.hidden_size).to(x.device), 
                torch.zeros(batch_size, self.hidden_size).to(x.device))
      hidden = (hx.unsqueeze(0), cx.unsqueeze(0)) # compatibility hack
    
    for t in range(seq_size):
      x_t = x[t, :, :]
      hidden = self.ResidualLSTMCell(x_t, hidden, w_ih=self.w_ih, w_hh=self.w_hh, 
                                     b_ih=self.b_ih, b_hh=self.b_hh)
      output.append(hidden[0])
    output = torch.cat(output).view(seq_size, batch_size, self.hidden_size)
    return output, hidden



class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dilations, dropout=0, cell_type='GRU', batch_first=False):

        super(DRNN, self).__init__()

        self.dilations = dilations
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "ResidualLSTM":
            cell = ResidualLSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size,
                                                       hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM' or self.cell_type == 'ResidualLSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden) # compatibility hack

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, autograd.Variable(zeros_)))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = autograd.Variable(torch.zeros(batch_size, hidden_dim))
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM" or self.cell_type == 'ResidualLSTM':
            memory = autograd.Variable(torch.zeros(batch_size, hidden_dim))
            if use_cuda:
                memory = memory.cuda()
            return hidden, memory
        else:
            return hidden


if __name__ == '__main__':
    n_inp = 10
    n_hidden = 16
    n_layers = 3
    batch_size = 100
    n_windows = 5

    model = DRNN(n_inp, n_hidden, n_layers, cell_type='ResidualLSTM', dilations=[1,2])

    test_x1 = torch.autograd.Variable(torch.randn(n_windows, batch_size, n_inp))
    test_x2 = torch.autograd.Variable(torch.randn(n_windows, batch_size, n_inp))

    out, hidden = model(test_x1)