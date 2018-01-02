import glob
import os
import cv2
import h5py
import numpy as np

import pdb

def extract_frames(video_dir, frame_dir):

    duration = 16   # 16 frames for 3d cnn

    for video in glob.glob(video_dir + '/*.mp4'):
        tokens = str(video).split('/')
        filename = (tokens[-1].split('.'))[0]

        # extract frames per video
        vidcap = cv2.VideoCapture(video)    # major version of cv >= 3
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # delete the last several frames (< 16 frames)
        num_frames -= (num_frames % duration)
        cnt = 0
        clips = None
        frames = None
        while vidcap.isOpened() and cnt < num_frames:
            success, image = vidcap.read()
            if success:
                print(os.path.join(filename, '%d.png') % cnt)
                image = cv2.resize(image, (112, 112))
                if cnt % duration == 0:
                    frames = image.reshape((1, -1, 112, 112))
                else:
                    frames = np.vstack((frames, image.reshape(1, -1, 112, 112)))
                # already 16 frames
                if (cnt + 1) % duration == 0:
                    frames = frames.transpose(1, 0, 2, 3)  # ch, d, h, w
                    frames = np.expand_dims(frames, axis = 0)  # expand batch axis
                    if clips is not None:
                        clips = np.vstack((clips, frames))
                    else:
                        clips = frames
                cnt += 1
            else:
                break

        cv2.destroyAllWindows()
        vidcap.release()

        # write clips (per video) to h5 file
        h5 = h5py.File(os.path.join(frame_dir, filename + '.h5'))
        h5.create_dataset('clips', data = clips)    # b, ch, d, h, w
        h5.close()