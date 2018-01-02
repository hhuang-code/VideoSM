import csv
import glob
import math
import h5py
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from more_itertools import chunked

import pdb

def plot_scores(filename, scores):

    plt.plot(np.linspace(0, len(scores), num = len(scores)), scores)
    # axis ranges
    plt.axis([0, len(scores), 0, 5])
    plt.title(filename)
    plt.xlabel('frame index')
    plt.ylabel('scores')
    plt.show()

def create_tvsum_gt(gt_src_file, gt_dest_file, sum_rate):

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
                avg_scores = np.array([])
            # every frame has a score
            scores = [float(s) for s in row[2].strip().split(',')]
            # stack frame scores
            if avg_scores.size == 0:
                avg_scores = np.hstack((avg_scores, scores))
            else:
                avg_scores = np.vstack((avg_scores, scores))
            cnt += 1
            if cnt == 20:   # calculate avg scores for a video
                avg_scores = np.mean(avg_scores, axis = 0)
                # delete the last several frames (< 16 frames)
                if len(avg_scores) % duration != 0:
                    avg_scores = avg_scores[: -(len(avg_scores) % duration)]
                # calculate avg scores for every 16 frames
                avg_scores = [sum(x) / len(x) for x in chunked(avg_scores, duration)]
                # to plot avg scores, uncomment it
                # plot_scores(filename, avg_scores)
                sorted_scores = sorted(avg_scores, reverse = True)  # descending order
                threshold = sorted_scores[int(math.floor(len(avg_scores) * sum_rate)) - 1]
                gt_scores = [int(x >= threshold) for x in avg_scores]
                # add gt_scores to gt_dict
                gt_dict[filename] = gt_scores
                cnt = 0

    # write gt_dict to h5 file
    h5 = h5py.File(gt_dest_file)
    for k, v in gt_dict.items():
        h5.create_dataset(k, data = v)
    h5.close()

def create_summe_gt(gt_src_dir, gt_dest_file, sum_rate):

    duration = 16   # 16 frames for 3d cnn
    gt_dict = {}    # store ground-truth for this datase

    for matfile in glob.glob(gt_src_dir + '/*.mat'):
        filename = matfile[matfile.rfind('/') + 1 : -4]
        mat = spio.loadmat(matfile, squeeze_me = True)
        avg_scores = mat['gt_score']
        max_score = max(avg_scores)
        min_score = min(avg_scores)
        # scale to 0.0 ~ 5.0
        avg_scores = 5 * (avg_scores - min_score) / (max_score - min_score)
        # delete the last several frames (< 16 frames)
        if len(avg_scores) % duration != 0:
            avg_scores = avg_scores[: -(len(avg_scores) % duration)]
        # calculate avg scores for every 16 frames
        avg_scores = [sum(x) / len(x) for x in chunked(avg_scores, duration)]
        # to plot avg scores, uncomment it
        # plot_scores(filename, avg_scores)
        sorted_scores = sorted(avg_scores, reverse=True)  # descending order
        threshold = sorted_scores[int(math.floor(len(avg_scores) * sum_rate)) - 1]
        gt_scores = [int(x >= threshold) for x in avg_scores]
        # add gt_scores to gt_dict
        gt_dict[filename] = gt_scores

    # write gt_dict to h5 file
    h5 = h5py.File(gt_dest_file)
    for k, v in gt_dict.items():
        h5.create_dataset(k, data = v)
    h5.close()