import torch
import torch.autograd as autograd

from networks.stdatt_lstm import *
from networks.t_cnn import *

import pdb

if __name__ == '__main__':

    """
    # test attlstm

    input_size = 4
    hidden_size = 3

    inputs = [autograd.Variable(torch.randn((1, input_size))) for _ in range(5)]
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)

    pdb.set_trace()

    attlstm = AttLstm(input_size, hidden_size) 

    output = attlstm(inputs) 

    print(output)
    """

    # test tcnn
    
    torch.manual_seed(1)
    inputs = Variable(torch.randn(1, 3, 32, 32))

    pdb.set_trace()

    inputs = inputs.expand(1, 5, -1, -1, -1)
    inputs = torch.transpose(inputs, 1, 2)

    tcnn = TemporalCNN()

    output = tcnn(inputs)

    print(output)
