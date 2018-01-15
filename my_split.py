from tools.split_data import *
from config import *

import pdb

"""
split dataset to training set and testing set
"""
if __name__ == '__main__':

    config = get_config(mode = 'train')

    # return values are lists
    train, test = get_datasets(config.combined_dir, config.gt_dir, trainset = 4, testset = 1)

    print(train)

    print(test)