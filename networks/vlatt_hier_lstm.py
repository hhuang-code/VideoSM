import math
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import pdb

"""
Variable-length attention mechanism in hierarchical LSTM
"""

class VLAttHierLstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(VLAttHierLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # default maximum upgrade length
        self.up_len = 80

        # bottom layer of LSTM
        self.bot_lstm = nn.LSTM(input_size, hidden_size, num_layers)

        # top layer of LSTM
        self.top_lstm = nn.LSTM(hidden_size, hidden_size, num_layers)

        # affine transformation for lstm hidden state
        self.att_hw = nn.Linear(hidden_size, hidden_size)

        # affine transformation for context
        self.att_cw = nn.Linear(hidden_size, hidden_size)

        # attention bias
        self.att_bias = nn.Parameter(torch.zeros(hidden_size))

        # affine transformation for vector to scalar
        self.att_v2s = nn.Linear(hidden_size, 1)

        # initial states
        self.h_0 = autograd.Variable(torch.zeros(1, 1, hidden_size)).cuda()
        self.c_0 = autograd.Variable(torch.zeros(1, 1, hidden_size)).cuda()

    def forward(self, x):
        bot_lstm_out, (bot_h_n, bot_c_n) = self.bot_lstm(x, (self.h_0, self.c_0))

        output = None   # output of VLAttHierLstm

        seq_len = len(x)

        # set upgrade length
        up_len = min(self.up_len, math.floor(math.sqrt(seq_len)))
        # evenly spaced index
        idx = np.linspace(up_len - 1, math.pow(up_len, 2) - 1, num = up_len)
        # input for top lstm
        up_x = torch.cat([bot_lstm_out[i] for i in idx])
        # append the last output of bottom lstm
        if idx[-1] != seq_len - 1:
            up_x = torch.cat((up_x, bot_lstm_out[-1]))

        top_lstm_out, (top_h_n, top_c_n) = self.top_lstm(up_x.view(len(up_x), 1, -1), (self.h_0, self.c_0))

        ctx = top_h_n   # context
        for i in range(seq_len):
            y = bot_lstm_out[: (i + 1)]
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
