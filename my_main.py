import torch
import torch.autograd as autograd

from networks.stdatt_lstm import *

from tools.split_data import *
from config import *
from model import *

import pdb

if __name__ == '__main__':

    config = get_config(mode = 'train')

    # get training and testing set; return values are lists
    train_set, test_set = get_datasets(config.combined_dir, config.gt_dir, trainset = 4, testset = 1)

    model = Model(config, train_set, test_set)

    model.build()

    model.train()

    model.test()

