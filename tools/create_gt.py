import os
import re
import cv2
import csv
import glob
import math
import h5py
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from more_itertools import chunked
from collections import Counter

from networks.resnet import *
from cpd_auto import cpd_auto

import pdb

def plot_scores(filename, scores):

    plt.plot(np.linspace(0, len(scores), num = len(scores)), scores)
    # axis ranges
    plt.axis([0, len(scores), 0, 5])
    plt.title(filename)
    plt.xlabel('frame/clip index')
    plt.ylabel('scores')
    plt.show()

def create_tvsum_gt(gt_src_file, gt_dest_file):

    duration = 16   # 16 frames for 3d cnn
    gt_dict = {}    # store ground-truth for this dataset

    with open(gt_src_file) as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        # every 20 lines correspond to one video
        cnt = 0
        filename = None
        avg_scores = np.array([])
        for row in reader:
            if cnt == 0:    # next video
                filename = row[0]
                avg_scores = None
            # every frame has a score
            scores = [float(s) for s in row[2].strip().split(',')]
            # stack frame scores
            if avg_scores is None:
                avg_scores = scores
            else:
                avg_scores = np.vstack((avg_scores, scores))
            cnt += 1
            if cnt == 20:   # calculate avg scores for a video
                counters = [Counter(avg_scores[:, i]) for i in range(avg_scores.shape[1])]
                most_scores = [counters[i].most_common(1)[0][0] for i in range(len(counters))]
                avg_scores = np.mean(avg_scores, axis = 0)
                # # scale to 1.0 ~ 5.0
                # max_score = max(avg_scores)
                # min_score = min(avg_scores)
                # avg_scores = 4 * (avg_scores - min_score) / (max_score - min_score) + 1
                # # delete the last several frames (< 16 frames)
                # if len(avg_scores) % duration != 0:
                #     avg_scores = avg_scores[: -(len(avg_scores) % duration)]
                # # calculate avg scores for every 16 frames
                # clip_scores = [float(sum(x)) / len(x) for x in chunked(avg_scores, duration)]
                # clip_scores = np.array([round(x) for x in clip_scores])
                # to plot avg scores, uncomment it
                if filename == '-esJrBWj2d8':
                    plot_scores(filename, avg_scores)
                    plot_scores(filename, most_scores)
                # add clip_scores to gt_dict
                gt_dict[filename] = avg_scores
                cnt = 0

    # write gt_dict to h5 file
    # h5 = h5py.File(gt_dest_file)
    # for k, v in gt_dict.items():
    #     h5.create_dataset(k, data = v)
    # h5.close()

def create_summe_gt(gt_src_dir, gt_dest_file):

    duration = 16   # 16 frames for 3d cnn
    gt_dict = {}    # store ground-truth for this datase

    for matfile in glob.glob(gt_src_dir + '/*.mat'):
        filename = matfile[matfile.rfind('/') + 1 : -4]
        mat = spio.loadmat(matfile, squeeze_me = True)
        avg_scores = mat['gt_score']
        # scale to 1.0 ~ 5.0
        max_score = max(avg_scores)
        min_score = min(avg_scores)
        avg_scores = 4 * (avg_scores - min_score) / (max_score - min_score) + 1
        # delete the last several frames (< 16 frames)
        if len(avg_scores) % duration != 0:
            avg_scores = avg_scores[: -(len(avg_scores) % duration)]
        # calculate avg scores for every 16 frames
        clip_scores = [sum(x) / len(x) for x in chunked(avg_scores, duration)]
        clip_scores = np.array([round(x) for x in clip_scores])
        # to plot avg scores, uncomment it
        # plot_scores(filename, clip_scores)
        # add clip_scores to gt_dict
        gt_dict[filename] = clip_scores

    # write gt_dict to h5 file
    h5 = h5py.File(gt_dest_file)
    for k, v in gt_dict.items():
        h5.create_dataset(k, data = v)
    h5.close()

"""
Args:
    dtype: 'youtube' or 'openvideo'
"""
def create_youtube_gt(video_dir, gt_src_dir, gt_dest_file, dtype):

    duration = 16  # 16 frames for 3d cnn
    gt_dict = {}  # store ground-truth for this datase

    # build resnet class
    resnet = ResNet()
    new_size = (224, 224)

    if dtype == 'youtube':
        video_path = video_dir + '/*.avi'
        key_appendix = '.jpg'
        regex = r'(frame)(\d+)(\.jpg)'  # frame#.jpg
    elif dtype == 'openvideo':
        video_path = video_dir + '/*.mpg'
        key_appendix = '.jpeg'
        regex = r'(Frame)(\d+)(\.jpeg)'  # Frame#.jpeg
    else:
        raise Exception('No such video dataset')

    pdb.set_trace()

    for video in glob.glob(video_path):
        tokens = str(video).split('/')
        filename = (tokens[-1].split('.'))[0]
        video_fea = None    # all frame features

        # extract frame features (resnet101) per video
        vidcap = cv2.VideoCapture(video)  # major version of cv >= 3
        cnt = 0
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                print(os.path.join(filename, '%d.png') % cnt)
                image = cv2.resize(image, new_size)
                res_pool5 = resnet(image)
                # gpu variable -> cpu variable -> tensor -> numpy array -> 1D array
                frame_fea = res_pool5.cpu().data.numpy().flatten()
                if video_fea is not None:
                    video_fea = np.vstack((video_fea, frame_fea))
                else:
                    video_fea = frame_fea
                cnt += 1
            else:
                break

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.destroyAllWindows()
        vidcap.release()

        # segment video
        length = num_frames / fps     # in second
        kernel = np.dot(video_fea, video_fea.T)
        m = int(math.ceil(length / 2.0))  # maximum change points, each segment is about 2s
        cps, _ = cpd_auto(kernel, m, 1)

        cps = np.concatenate(([0], cps, [num_frames - 1]))

        # read manual annotation
        avg_scores = np.zeros(num_frames) + 1
        for img in glob.glob(gt_src_dir + '/' + filename + '/*/*' + key_appendix):
            if re.search(regex, img):
                idx = int(re.search(regex, img).group(2))
                ks = fps * (idx - 1) + 1    # start frame idx
                ke = fps * idx              # end frame idx
                # find maximum overlap with cps
                maxlap = 0
                mcs = 0
                mce = 0
                for i in range(len(cps) - 1):
                    cs = cps[i]             # current start frame idx
                    ce = cps[i + 1]         # current end frame idx
                    overlap = max(0, min(ce, ke) - max(cs, ks))     # calculate overlap
                    if overlap > maxlap:
                        maxlap = overlap
                        mcs = cs
                        mce = ce
                # record scores
                avg_scores[mcs : (mce + 1)] += 1
            else:
                continue

        # scale to 1.0 ~ 5.0
        max_score = max(avg_scores)
        min_score = min(avg_scores)
        avg_scores = 4 * (avg_scores - min_score) / (max_score - min_score) + 1
        # delete the last several frames (< 16 frames)
        if len(avg_scores) % duration != 0:
            avg_scores = avg_scores[: -(len(avg_scores) % duration)]
        # calculate avg scores for every 16 frames
        clip_scores = [sum(x) / len(x) for x in chunked(avg_scores, duration)]
        clip_scores = np.array([round(x) for x in clip_scores])
        # to plot avg scores, uncomment it
        plot_scores(filename, clip_scores)
        gt_dict[filename] = clip_scores

    # write gt_dict to h5 file
    h5 = h5py.File(gt_dest_file)
    for k, v in gt_dict.items():
        h5.create_dataset(k, data = v)
    h5.close()


