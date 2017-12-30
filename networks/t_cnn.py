import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pdb

class TemporalCNN(nn.Module):

    def __init__(self):
        super(TemporalCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 4, kernel_size = (3, 3, 3))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

    def forward(self, x):

        pdb.set_trace()

        x = self.conv1(x)
        out = self.pool1(x)

        return out
