import argparse
from pathlib import Path

"""
set configuration arguments as class attributes
"""
class Config(object):

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

"""
get configuration arguments
"""
def get_config(**kwargs):

    parser = argparse.ArgumentParser()

    # lstm parameters
    parser.add_argument('--input_size', type = int, default = 4096)
    parser.add_argument('--hidden_size', type = int, default = 1024)
    parser.add_argument('--num_layers', type = int, default = 1)

    # training parameters
    parser.add_argument('--max_epoch_num', type = int, default = 1)
    parser.add_argument('--learning_rate', type = float, default = 0.0001)
    parser.add_argument('--weight_decay', type = float, default = 0.005)

    # dataset path
    # combined dir: data combining video and ground-truth
    parser.add_argument('--video_dir_tvsum', type = str, default = Path('/localdisk/videosm/dataset/video/TVSum'))
    parser.add_argument('--gt_dir_tvsum', type = str, default = Path('/localdisk/videosm/dataset/gt/TVSum'))
    parser.add_argument('--combined_dir_tvsum', type = str, default = Path('/localdisk/videosm/dataset/combined/TVSum'))
   
    parser.add_argument('--video_dir_summe', type = str, default = Path('/localdisk/videosm/dataset/video/SumMe'))
    parser.add_argument('--gt_dir_summe', type = str, default = Path('/localdisk/videosm/dataset/gt/SumMe'))
    parser.add_argument('--combined_dir_summe', type=str, default=Path('/localdisk/videosm/dataset/combined/SumMe'))

    parser.add_argument('--video_dir_youtube', type=str, default=Path('/localdisk/videosm/dataset/video/Youtube'))
    parser.add_argument('--gt_dir_youtube', type=str, default=Path('/localdisk/videosm/dataset/gt/Youtube'))
    parser.add_argument('--combined_dir_youtube', type=str, default=Path('/localdisk/videosm/dataset/combined/Youtube'))
   
    parser.add_argument('--video_dir_openvideo', type = str, default = Path('/localdisk/videosm/dataset/video/OpenVideo'))
    parser.add_argument('--gt_dir_openvideo', type = str, default = Path('/localdisk/videosm/dataset/gt/OpenVideo'))
    parser.add_argument('--combined_dir_openvideo', type = str, default = Path('/localdisk/videosm/dataset/combined/OpenVideo'))

    parser.add_argument('--gt_dir', type=str, default=Path('/localdisk/videosm/dataset/gt'))
    parser.add_argument('--combined_dir', type = str, default = Path('/localdisk/videosm/dataset/combined'))
    
    # model path
    parser.add_argument('--c3d_model_dir', type = str, default = Path('/localdisk/videosm/model/'))
    parser.add_argument('--videosm_model_dir', type = str, default = Path('/localdisk/videosm/model/'))
    
    args = parser.parse_args()

    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)
