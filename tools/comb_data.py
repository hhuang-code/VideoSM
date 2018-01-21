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

from networks.resnet import *
from helpers import *
from cpd_auto import cpd_auto

import pdb

def plot_scores(filename, scores):

    plt.plot(np.linspace(0, len(scores), num = len(scores)), scores)
    # axis ranges
    plt.axis([0, len(scores), 0, 1.2 * max(scores)])
    plt.title(filename)
    plt.xlabel('frame/clip index')
    plt.ylabel('scores')
    plt.show()

"""
convert floating point scores to 0-1 scores
"""
def convert_bin(clip_scores, sum_rate):
    clip_scores = [x * 1000 for x in clip_scores]   # up scale
    clip_scores = [int(round(x)) for x in clip_scores]

    n = len(clip_scores)
    W = int(n * sum_rate)
    val = clip_scores
    wt = [1 for x in range(n)]
    _, sln = knapsack(W, wt, val, n)

    return sln[n][W]

"""
create ground-truth for TVSum
"""
def create_tvsum_gt(gt_src_file, sum_rate = 0.15):

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
                avg_scores = np.mean(avg_scores, axis = 0)
                # # scale to 0.0 ~ 1.0
                max_score = max(avg_scores)
                min_score = min(avg_scores)
                avg_scores = (avg_scores - min_score) / (max_score - min_score)
                # delete the last several frames (< 16 frames)
                if len(avg_scores) % duration != 0:
                    avg_scores = avg_scores[: -(len(avg_scores) % duration)]
                # calculate avg scores for every 16 frames
                clip_scores = [float(sum(x)) / len(x) for x in chunked(avg_scores, duration)]
                # from floating point scores to binary scores
                key_idx = convert_bin(clip_scores, sum_rate)
                clip_bins = np.zeros(len(clip_scores))
                for x in key_idx:
                    clip_bins[x] = 1

                # to plot avg scores, uncomment it
                # plot_scores(filename, avg_scores)
                # plot_scores(filename, clip_scores)
                # plot_scores(filename, clip_bins)

                # add clip_scores to gt_dict
                gt_dict[filename] = clip_scores
                cnt = 0

    return gt_dict

"""
create ground-truth for SumMe
"""
def create_summe_gt(gt_src_dir, sum_rate = 0.15):

    duration = 16   # 16 frames for 3d cnn
    gt_dict = {}    # store ground-truth for this datase

    for matfile in glob.glob(gt_src_dir + '/*.mat'):
        filename = matfile[matfile.rfind('/') + 1 : -4]
        mat = spio.loadmat(matfile, squeeze_me = True)
        avg_scores = mat['gt_score']

        # SPECIAL CASE: delete the last frame to keep the number of ground-truth the same as video frames extract by opencv
        if filename == 'playing_ball':
            avg_scores = avg_scores[:-2]

        # scale to 0.0 ~ 1.0
        max_score = max(avg_scores)
        min_score = min(avg_scores)
        avg_scores = (avg_scores - min_score) / (max_score - min_score)
        # delete the last several frames (< 16 frames)
        if len(avg_scores) % duration != 0:
            avg_scores = avg_scores[: -(len(avg_scores) % duration)]
        # calculate avg scores for every 16 frames
        clip_scores = [sum(x) / len(x) for x in chunked(avg_scores, duration)]
        # from floating point scores to binary scores
        key_idx = convert_bin(clip_scores, sum_rate)
        clip_bins = np.zeros(len(clip_scores))
        for x in key_idx:
            clip_bins[x] = 1

        # to plot avg scores, uncomment it
        # plot_scores(filename, avg_scores)
        # plot_scores(filename, clip_scores)
        # plot_scores(filename, clip_bins)

        # add clip_scores to gt_dict
        gt_dict[filename] = clip_scores

    return gt_dict

"""
create ground-truth for Youtube
"""
def create_youtube_gt(video_dir, gt_src_dir, sum_rate = 0.15):

    duration = 16  # 16 frames for 3d cnn
    gt_dict = {}  # store ground-truth for this datase

    # build resnet class
    resnet = ResNet()
    new_size = (224, 224)

    video_path = video_dir + '/*.avi'
    regex = r'(frame)(\d+)(\.jpg)'  # frame#.jpg

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
                if video_fea is None:
                    video_fea = frame_fea
                else:
                    video_fea = np.vstack((video_fea, frame_fea))
                cnt += 1
            else:
                break

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.destroyAllWindows()
        vidcap.release()

        # some frames cannot be read
        num_frames = min(num_frames, cnt)

        # segment video
        length = num_frames / fps     # in second
        kernel = np.dot(video_fea, video_fea.T)
        m = int(math.ceil(length / 2.0))  # maximum change points, each segment is about 2s
        cps, _ = cpd_auto(kernel, m, 1)

        cps = np.concatenate(([0], cps, [num_frames - 1]))

        # read manual annotation
        avg_scores = np.zeros(num_frames)
        for img in glob.glob(gt_src_dir + '/' + filename + '/*/*' + '.jpg'):
            if re.search(regex, img):
                idx = int(re.search(regex, img).group(2))   # second-base
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

        # scale to 0.0 ~ 1.0
        max_score = max(avg_scores)
        min_score = min(avg_scores)
        avg_scores = (avg_scores - min_score) / (max_score - min_score)
        # delete the last several frames (< 16 frames)
        if len(avg_scores) % duration != 0:
            avg_scores = avg_scores[: -(len(avg_scores) % duration)]
        # calculate avg scores for every 16 frames
        clip_scores = [sum(x) / len(x) for x in chunked(avg_scores, duration)]
        # from floating point scores to binary scores
        key_idx = convert_bin(clip_scores, sum_rate)
        clip_bins = np.zeros(len(clip_scores))
        for x in key_idx:
            clip_bins[x] = 1

        # to plot avg scores, uncomment it
        # plot_scores(filename, avg_scores)
        # plot_scores(filename, clip_scores)
        # plot_scores(filename, clip_bins)

        gt_dict[filename] = clip_scores

    return gt_dict

"""
create ground-truth for OpenVideo
"""
def create_openvideo_gt(video_dir, gt_src_dir, gt_dest_file, sum_rate = 0.15):

    duration = 16  # 16 frames for 3d cnn
    gt_dict = {}  # store ground-truth for this datase

    # build resnet class
    resnet = ResNet()
    new_size = (224, 224)

    video_path = video_dir + '/*.mpg'
    regex = r'(Frame)(\d+)(\.jpeg)'  # Frame#.jpeg

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

        # some frames cannot be read
        num_frames = min(num_frames, cnt)

        # segment video
        length = num_frames / fps     # in second
        kernel = np.dot(video_fea, video_fea.T)
        m = int(math.ceil(length / 2.0))  # maximum change points, each segment is about 2s
        cps, _ = cpd_auto(kernel, m, 1)

        cps = np.concatenate(([0], cps, [num_frames - 1]))

        # read manual annotation
        avg_scores = np.zeros(num_frames)
        for img in glob.glob(gt_src_dir + '/' + filename + '/*/*' + '.jpeg'):
            if re.search(regex, img):
                idx = int(re.search(regex, img).group(2))   # frame-base
                ks = idx - int(fps / 2)         # start frame idx
                ke = idx + int(fps / 2) + 1     # end frame idx
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

        # scale to 0.0 ~ 1.0
        max_score = max(avg_scores)
        min_score = min(avg_scores)
        avg_scores = (avg_scores - min_score) / (max_score - min_score)
        # delete the last several frames (< 16 frames)
        if len(avg_scores) % duration != 0:
            avg_scores = avg_scores[: -(len(avg_scores) % duration)]
        # calculate avg scores for every 16 frames
        clip_scores = [sum(x) / len(x) for x in chunked(avg_scores, duration)]
        # from floating point scores to binary scores
        key_idx = convert_bin(clip_scores, sum_rate)
        clip_bins = np.zeros(len(clip_scores))
        for x in key_idx:
            clip_bins[x] = 1

        # to plot avg scores, uncomment it
        # plot_scores(filename, avg_scores)
        # plot_scores(filename, clip_scores)
        # plot_scores(filename, clip_bins)

        gt_dict[filename] = clip_scores

    return gt_dict

"""
combine video data and ground-truth together for TVSum
"""
def combine_tvsum(video_dir, gt_src_file, combined_dir, sum_rate):

    # get converted ground-truth
    gt_dict = create_tvsum_gt(gt_src_file, sum_rate)

    # get frames for every video
    for filename, gt in gt_dict.iteritems():

        if not os.path.exists(os.path.join(combined_dir, filename + '.h5')):
            clips = get_frames(video_dir, filename + '.mp4')
            # clips and ground-truth have the same length
            assert len(clips) == len(gt), 'clips and ground-truth have different length!'

            # write combined data (per video) to h5 file
            h5 = h5py.File(os.path.join(combined_dir, filename + '.h5'))
            # h5.create_dataset('data', data = clips, compression = 'gzip', compression_opts = 9)
            # h5.create_dataset('gt', data = gt_dict[filename], compression = 'gzip', compression_opts = 9)
            h5.create_dataset('video', data = os.path.join(video_dir, filename + '.mp4'))
            h5.create_dataset('gt', data = gt_dict[filename])
            h5.close()

"""
combine video data and ground-truth together for Summe
"""
def combine_summe(video_dir, gt_src_dir, combined_dir, sum_rate):

    # get converted ground-truth
    gt_dict = create_summe_gt(gt_src_dir, sum_rate)

    # get frames for every video
    for filename, gt in gt_dict.iteritems():

        if not os.path.exists(os.path.join(combined_dir, filename + '.h5')):
            clips = get_frames(video_dir, filename + '.mp4')
            # clips and ground-truth have the same length
            assert len(clips) == len(gt), 'clips and ground-truth have different length!'

            # write combined data (per video) to h5 file
            h5 = h5py.File(os.path.join(combined_dir, filename + '.h5'))
            # h5.create_dataset('data', data = clips, compression = 'gzip', compression_opts = 9)
            # h5.create_dataset('gt', data = gt_dict[filename], compression = 'gzip', compression_opts = 9)
            h5.create_dataset('video', data = os.path.join(video_dir, filename + '.mp4'))
            h5.create_dataset('gt', data = gt_dict[filename])
            h5.close()

"""
combine video data and ground-truth together for Youtube
"""
def combine_youtube(video_dir, gt_src_dir, combined_dir, sum_rate):

    gt_dict = {}

    if not os.path.exists(os.path.join(gt_src_dir, 'gt.h5')):
        # get converted ground-truth
        gt_dict = create_youtube_gt(video_dir, gt_src_dir, sum_rate)
        gt_h5 = h5py.File(os.path.join(gt_src_dir, 'gt.h5'), 'w')
        for k, v in gt_dict.items():
            gt_h5.create_dataset(k, data = v)
        gt_h5.close()
    else:
        # load converted ground-truth
        gt_h5 = h5py.File(os.path.join(gt_src_dir, 'gt.h5'), 'r')
        for key in gt_h5.keys():
            gt_dict[key] = gt_h5[key]
        gt_h5.close()

    # get frames for every video
    for filename, gt in gt_dict.iteritems():

        if not os.path.exists(os.path.join(combined_dir, filename + '.h5')):
            clips = get_frames(video_dir, filename + '.avi')
            # clips and ground-truth have the same length
            assert len(clips) == len(gt), 'clips and ground-truth have different length!'

            # write combined data (per video) to h5 file
            h5 = h5py.File(os.path.join(combined_dir, filename + '.h5'))
            # h5.create_dataset('data', data = clips, compression = 'gzip', compression_opts = 9)
            # h5.create_dataset('gt', data = gt_dict[filename], compression = 'gzip', compression_opts = 9)
            h5.create_dataset('video', data = os.path.join(video_dir, filename + '.avi'))
            h5.create_dataset('gt', data = gt_dict[filename])
            h5.close()

    gt_h5.close()

"""
combine video data and ground-truth together for OpenVideo
"""
def combine_openvideo(video_dir, gt_src_dir, combined_dir, sum_rate):

    gt_dict = {}

    if not os.path.exists(os.path.join(gt_src_dir, 'gt.h5')):
        # get converted ground-truth
        gt_dict = create_openvideo_gt(video_dir, gt_src_dir, sum_rate)
        gt_h5 = h5py.File(os.path.join(gt_src_dir, 'gt.h5'), 'w')
        for k, v in gt_dict.items():
            gt_h5.create_dataset(k, data = v)
    else:
        # load converted ground-truth
        gt_h5 = h5py.File(os.path.join(gt_src_dir, 'gt.h5'), 'r')
        for key in gt_h5.keys():
            gt_dict[key] = gt_h5[key]

    # get frames for every video
    for filename, gt in gt_dict.iteritems():

        if not os.path.exists(os.path.join(combined_dir, filename + '.h5')):
            clips = get_frames(video_dir, filename + '.mpg')
            # clips and ground-truth have the same length
            assert len(clips) == len(gt), 'clips and ground-truth have different length!'

            # write combined data (per video) to h5 file
            h5 = h5py.File(os.path.join(combined_dir, filename + '.h5'))
            # h5.create_dataset('data', data = clips, compression = 'gzip', compression_opts = 9)
            # h5.create_dataset('gt', data = gt_dict[filename], compression = 'gzip', compression_opts = 9)
            h5.create_dataset('video', data = os.path.join(video_dir, filename + '.mpg'))
            h5.create_dataset('gt', data = gt_dict[filename])
            h5.close()

    gt_h5.close()