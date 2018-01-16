import torch
import torch.autograd as autograd
import torch.nn as nn

from networks.stdatt_lstm import *
from networks.vlatt_lstm import *
from networks.vlatt_hier_lstm import *

import pdb

if __name__ == '__main__':

    # test attlstm

    input_size = 4
    hidden_size = 3

    input = [autograd.Variable(torch.randn((1, input_size))) for _ in range(900)]
    input = torch.cat(input).view(len(input), 1, -1)

    pdb.set_trace()

    # stdattlstm = StdAttLstm(input_size, hidden_size)
    # output = stdattlstm(input)

    # vlattlstm = VLAttLstm(input_size, hidden_size)
    # output = vlattlstm(input)

    vlatthierlstm = VLAttHierLstm(input_size, hidden_size)
    output = vlatthierlstm(input)

    print(output)
