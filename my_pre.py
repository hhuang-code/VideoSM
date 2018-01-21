import os
import pandas as pd

from tools.helpers import *
from config import *

import pdb

"""
get basic video information and down sample videos
"""
def get_video_info(config):

    columns = ['fps', 'num_frames', 'duration']
    df = pd.DataFrame(columns = columns)

    # TVSum
    video_dir = str(config.video_dir_tvsum)
    video_path = video_dir + '/*.mp4'
    for video in glob.glob(video_path):
        filename = video[video.rfind('/') + 1: len(video)]
        fps, num_frames, duration = video_stats(video_dir, filename)
        df.loc[os.path.join(video_dir, filename)] = [fps, num_frames, duration]

    # SumMe
    video_dir = str(config.video_dir_summe)
    video_path = video_dir + '/*.mp4'
    for video in glob.glob(video_path):
        filename = video[video.rfind('/') + 1: len(video)]
        fps, num_frames, duration = video_stats(video_dir, filename)
        df.loc[os.path.join(video_dir, filename)] = [fps, num_frames, duration]

    # Youtube
    video_dir = str(config.video_dir_youtube)
    video_path = video_dir + '/*.avi'
    for video in glob.glob(video_path):
        filename = video[video.rfind('/') + 1: len(video)]
        fps, num_frames, duration = video_stats(video_dir, filename)
        df.loc[os.path.join(video_dir, filename)] = [fps, num_frames, duration]

    # openvideo
    video_dir = str(config.video_dir_openvideo)
    video_path = video_dir + '/*.mpg'
    for video in glob.glob(video_path):
        filename = video[video.rfind('/') + 1: len(video)]
        fps, num_frames, duration = video_stats(video_dir, filename)
        df.loc[os.path.join(video_dir, filename)] = [fps, num_frames, duration]

    df.to_csv(os.path.join(config.dataset_dir, 'video_stats.csv'))

def downsample(config):
    # TVSum
    video_dir = str(config.video_dir_tvsum)
    video_ds_dir = str(config.video_ds_dir_tvsum)
    video_path = video_dir + '/*.mp4'
    for in_video in glob.glob(video_path):
        filename = in_video[in_video.rfind('/') + 1: len(in_video)]
        out_video = os.path.join(video_ds_dir, filename)

        vidcap = cv2.VideoCapture(in_video)  # major version of cv >= 3
        fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = vidcap.get(cv2.CV_CAP_PROP_FOURCC)
        frame_size = (vidcap.get(cv2.CV_CAP_PROP_FRAME_WIDTH), vidcap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))

        # new sample rate
        if fps % 6 == 0:
            interval = 6
        elif fps % 5 == 0:
            interval = 5
        else:
            raise Exception('No such fps: ' + in_video)

        vidwrite = cv2.VideoWriter(out_video, fourcc, fps / interval, frame_size)

        cnt = 0
        while vidcap.isOpened() and cnt < num_frames:
            success, image = vidcap.read()
            if success:
                if cnt % interval == 0:
                    print(os.path.join(filename, '%d.png') % cnt)
                    vidwrite.write(image)
                cnt += 1
            else:
                break

        vidcap.release()
        vidwrite.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    config = get_config(mode = 'train')

    # get_video_info(config)

    downsample(config)