from tools.convert import *
from config import *

import pdb

if __name__ == '__main__':

    config = get_config(mode = 'train')

    # extract frames in TVSum
    video_dir_tvsum = str(config.video_dir_tvsum)
    frame_dir_tvsum = str(config.frame_dir_tvsum)
    extract_frames(video_dir_tvsum, frame_dir_tvsum)

    # extract frames in TVSum
    gt_src_dir = str(config.gt_dir_summe) + '/GT'
    gt_dest_file = str(config.gt_dir_summe) + '/gt_summe.h5'
    #extract_frames(gt_src_dir, gt_dest_file, sum_rate)