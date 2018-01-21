import glob
import os
import cv2
import h5py
import numpy as np

import pdb

"""
extract frames from video
"""
def get_frames(video_dir, filename):

    duration = 16   # 16 frames for 3d cnn

    video = os.path.join(video_dir, filename)

    # extract video frames
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
            # resize for c3d
            image = cv2.resize(image, (112, 112))
            if cnt % duration == 0:
                frames = image.reshape((1, -1, 112, 112))
            else:
                frames = np.vstack((frames, image.reshape(1, -1, 112, 112)))
            # already 16 frames
            if (cnt + 1) % duration == 0:
                frames = frames.transpose(1, 0, 2, 3)  # ch, d, h, w
                frames = np.expand_dims(frames, axis = 0)  # expand batch axis
                if clips is None:
                    clips = frames
                else:
                    clips = np.vstack((clips, frames))
            cnt += 1
        else:
            break

    vidcap.release()
    cv2.destroyAllWindows()

    return clips    # b, ch, d, h, w

"""
get fps, the number of frames, and duration of a video
"""
def video_stats(video_dir, filename):

    video = os.path.join(video_dir, filename)

    print(video)

    vidcap = cv2.VideoCapture(video)    # major version of cv >= 3

    fps = vidcap.get(cv2.CAP_PROP_FPS)

    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = num_frames / fps

    cv2.destroyAllWindows()
    vidcap.release()

    return int(round(fps)), num_frames, int(round(duration))

"""
dynamic programming for 0-1 knapsack problem, based on https://www.geeksforgeeks.org/knapsack-problem/ 
Arg:
    W: maximum weight
    wt: weight (duration) list
    val: value (score) list
    n: length of wt (or val)
"""
def knapsack(W, wt, val, n):
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    sln = [[[] for x in range(W + 1)] for x in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                if val[i - 1] + K[i - 1][w - wt[i - 1]] > K[i - 1][w]:
                    K[i][w] = val[i - 1] + K[i - 1][w - wt[i - 1]]
                    sln[i][w] = list(sln[i - 1][w - wt[i - 1]])
                    if len(sln[i][w]) != 0:
                        sln[i][w].append(i - 1)
                    else:
                        sln[i][w] = [i - 1]
                else:
                    K[i][w] = K[i - 1][w]
                    sln[i][w] = list(sln[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
                sln[i][w] = list(sln[i - 1][w])

    return K, sln

# val = [4, 3, 6]
# wt = [1, 3, 5]
# W = 6
#
# opt, sln = knapsack(W, wt, val, len(val))