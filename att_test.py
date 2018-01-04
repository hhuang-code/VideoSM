import torch
import torch.autograd as autograd
import torch.nn as nn

from networks.att_lstm import *

import pdb

if __name__ == '__main__':

    # test attlstm

    input_size = 4
    hidden_size = 3

    inputs = [autograd.Variable(torch.randn((1, input_size))) for _ in range(5)]
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)

    attlstm = AttLstm(input_size, hidden_size)

    outputs = attlstm(inputs)

    print(outputs)
