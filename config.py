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

    # dataset path
    parser.add_argument('--video_dir_youtube', type = str, default = Path('/localdisk/videosm/dataset/video/Youtube'))
    parser.add_argument('--frame_dir_youtube', type = str, default = Path('/localdisk/videosm/dataset/frame/Youtube'))
    parser.add_argument('--feature_dir_youtube', type = str, default = Path('/localdisk/videosm/dataset/feature/Youtube'))
    parser.add_argument('--gt_dir_youtube', type = str, default = Path('/localdisk/videosm/dataset/gt/Youtube'))
  
    parser.add_argument('--video_dir_tvsum', type = str, default = Path('/localdisk/videosm/dataset/video/TVSum'))
    parser.add_argument('--frame_dir_tvsum', type = str, default = Path('/localdisk/videosm/dataset/frame/TVSum'))
    parser.add_argument('--feature_dir_tvsum', type = str, default = Path('/localdisk/videosm/dataset/feature/TVSum'))
    parser.add_argument('--gt_dir_tvsum', type = str, default = Path('/localdisk/videosm/dataset/gt/TVSum'))
   
    parser.add_argument('--video_dir_summe', type = str, default = Path('/localdisk/videosm/dataset/video/SumMe'))
    parser.add_argument('--frame_dir_summe', type = str, default = Path('/localdisk/videosm/dataset/frame/SumMe'))
    parser.add_argument('--feature_dir_summe', type = str, default = Path('/localdisk/videosm/dataset/feature/SumMe'))
    parser.add_argument('--gt_dir_summe', type = str, default = Path('/localdisk/videosm/dataset/gt/SumMe'))
   
    parser.add_argument('--video_dir_openvideo', type = str, default = Path('/localdisk/videosm/dataset/video/OpenVideo'))
    parser.add_argument('--frame_dir_openvideo', type = str, default = Path('/localdisk/videosm/dataset/frame/OpenVideo'))
    parser.add_argument('--feature_dir_openvideo', type = str, default = Path('/localdisk/videosm/dataset/feature/OpenVideo'))
    parser.add_argument('--gt_dir_openvideo', type = str, default = Path('/localdisk/videosm/dataset/gt/OpenVideo'))
    
    # model path
    parser.add_argument('--c3d_model_dir', type = str, default = Path('/localdisk/videosm/model/'))
    parser.add_argument('--videosm_model_dir', type = str, default = Path('/localdisk/videosm/model/'))
    
    args = parser.parse_args()

    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)
