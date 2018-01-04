import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import pdb

"""
Variable-length attention mechanism in LSTM
"""

class VLAttLstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(VLAttLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        # affine transformation for lstm hidden state
        self.att_hw = nn.Linear(hidden_size, hidden_size)

        # affine transformation for context
        self.att_cw = nn.Linear(hidden_size, hidden_size)

        # attention bias
        self.att_bias = nn.Parameter(torch.zeros(hidden_size))

        # affine transformation for vector to scalar
        self.att_v2s = nn.Linear(hidden_size, 1)

        # initial states
        self.h_0 = autograd.Variable(torch.zeros(1, 1, hidden_size))
        self.c_0 = autograd.Variable(torch.zeros(1, 1, hidden_size))

    def forward(self, x):

        lstm_out, (h_n, c_n) = self.lstm(x, (self.h_0, self.c_0))

        output = None   # output of VLAttLstm

        seq_len = len(x)
        ctx = h_n   # context
        for i in range(seq_len):
            pdb.set_trace()

            y = lstm_out[: (i + 1)]
            m = F.tanh(self.att_hw(y) + self.att_cw(ctx.expand(i + 1, -1, -1))
                       + self.att_bias.view(1, 1, -1).expand(i + 1, -1, -1))
            m = self.att_v2s(m)
            s = F.softmax(m, dim = 0)
            z = torch.sum(y * s, dim = 0).view(1, 1, -1)    # broadcasting when multiply y ans s
            if output is None:
                output = z
            else:
                output = torch.cat((output, z), dim = 0)

        return output
