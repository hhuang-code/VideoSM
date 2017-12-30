import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb

class tLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(tLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x, hidden = None):
        output, (h_n, c_n) = self.lstm(x, hidden)

        return output, (h_n, c_n)

input_size = 5
hidden_size = 4

torch.manual_seed(1)

tlstm = tLSTM(input_size, hidden_size, num_layers = 1)

inputs = [autograd.Variable(torch.randn((1, input_size))) for _ in range(5)]
inputs = torch.cat(inputs).view(len(inputs), 1, -1)

(h, c) = (autograd.Variable(torch.zeros(1, 1, hidden_size)),
    autograd.Variable(torch.zeros((1, 1, hidden_size))))

output, (h_n, c_n) = tlstm(inputs, (h, c))
print(output)

for i in inputs:
    out, (h, c) = tlstm(i.view(1, 1, -1), (h, c))
    print(out)
    print(h)

pdb.set_trace()

print()
