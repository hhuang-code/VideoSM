from tools.create_gt import *
from config import *

import pdb

if __name__ == '__main__':

    config = get_config(mode = 'train')

    sum_rate = 0.15

    # create ground-truth for TVSum
    gt_src_file = str(config.gt_dir_tvsum) + '/ydata-tvsum50-anno.tsv'
    gt_dest_file = str(config.gt_dir_tvsum) + '/gt_tvsum.h5'
    #create_tvsum_gt(gt_src_file, gt_dest_file, sum_rate)

    # create ground-truth for SumMe
    gt_src_dir = str(config.gt_dir_summe) + '/GT'
    gt_dest_file = str(config.gt_dir_summe) + '/gt_summe.h5'
    create_summe_gt(gt_src_dir, gt_dest_file, sum_rate)
