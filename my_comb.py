from tools.comb_data import *
from config import *

import pdb

"""
create (convert) ground-truth to suitable format for training and testing;
combine video data and converted ground-truth
"""
if __name__ == '__main__':

    config = get_config(mode = 'train')

    # create ground-truth for TVSum
    video_dir = str(config.video_dir_tvsum)
    gt_src_file = str(config.gt_dir_tvsum) + '/ydata-tvsum50-anno.tsv'
    combined_dir = str(config.combined_dir_tvsum)
    #combine_tvsum(video_dir, gt_src_file, combined_dir, sum_rate = 0.15)

    # create ground-truth for SumMe
    video_dir = str(config.video_dir_summe)
    gt_src_dir = str(config.gt_dir_summe) + '/GT'
    combined_dir = str(config.combined_dir_summe)
    combine_summe(video_dir, gt_src_dir, combined_dir, sum_rate = 0.15)

    # create ground-truth for Youtube
    video_dir = str(config.video_dir_youtube)
    gt_src_dir = str(config.gt_dir_youtube)
    combined_dir = str(config.combined_dir_youtube)
    #combine_youtube(video_dir, gt_src_dir, combined_dir, sum_rate = 0.15)

    # create ground-truth for OpenVideo
    video_dir = str(config.video_dir_openvideo)
    gt_src_dir = str(config.gt_dir_openvideo)
    combined_dir = str(config.combined_dir_openvideo)
    #combine_openvideo(video_dir, gt_src_dir, combined_dir, sum_rate = 0.15)