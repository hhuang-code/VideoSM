import torch
import torch.autograd as autograd
import torch.nn as nn

from networks.att_lstm import *
from networks.vl_att_lstm import *

import pdb

if __name__ == '__main__':

    # test attlstm

    input_size = 4
    hidden_size = 3

    input = [autograd.Variable(torch.randn((1, input_size))) for _ in range(5)]
    input = torch.cat(input).view(len(input), 1, -1)

    # attlstm = AttLstm(input_size, hidden_size)
    # output = attlstm(input)

    vlattlstm = VLAttLstm(input_size, hidden_size)
    output = vlattlstm(input)

    print(output)
