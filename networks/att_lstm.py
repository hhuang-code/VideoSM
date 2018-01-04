import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import pdb


class AttLstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(AttLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # encoder
        self.elstm = nn.LSTM(input_size, hidden_size, num_layers)

        # affine transformation for encoder lstm hidden state
        self.att_eh = nn.Linear(hidden_size, hidden_size)

        # affine transformation for decoder lstm hidden state (context)
        self.att_ch = nn.Linear(hidden_size, hidden_size)

        # affine transformation for vector to scalar
        self.att_v2s = nn.Linear(hidden_size, 1)

        # attention bias
        self.att_bias = nn.Parameter(torch.zeros(hidden_size))

        # decoder
        self.dlstm = nn.LSTM(hidden_size, hidden_size, num_layers)

        self.init_hidden = (autograd.Variable(torch.zeros(1, 1, hidden_size)),
                            autograd.Variable(torch.zeros(1, 1, hidden_size)))

    def forward(self, x):
        pdb.set_trace()

        e_output, (eh_n, ec_n) = self.elstm(x, self.init_hidden)

        seq_len = len(x)
        ctx = self.init_hidden[0]  # initial context
        c_n = self.init_hidden[1]
        out = None
        for i in range(seq_len):
            pdb.set_trace()
            m = F.tanh(self.att_eh(e_output) + self.att_ch(ctx.expand(seq_len, -1, -1))
                       + self.att_bias.view(1, 1, -1).expand(seq_len, -1, -1))
            m = self.att_v2s(m)
            s = F.softmax(m, dim = 0)
            z = torch.sum(e_output * s, dim = 0).view(1, 1, -1)  # broadcasting when multiply e_output and s
            d_output, (h_n, c_n) = self.dlstm(z, (ctx, c_n))
            ctx = h_n
            if i == 0:  # first time
                out = d_output
            else:
                out = torch.cat((out, d_output), dim = 0)

        return out
