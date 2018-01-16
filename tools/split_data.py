import pandas as pd
import numpy as np
import random
import os

import pdb

"""
split dataset into 'partitions' parts
"""
def ran_split(dataset, partitions = 5):

    parts=[]
    it_num = len(dataset) / partitions

    for _ in range(partitions - 1):
        part=[]
        new_dataset=[]
        part1 = random.sample(range(len(dataset)), int(it_num))
        for item in range(len(dataset)):
            if item in part1:
                part.append(dataset[item])
            else:
                new_dataset.append(dataset[item])

        dataset = new_dataset
        parts.append(part)

    parts.append(dataset)

    return parts    # len is 'partitions', and each partition has it_num elements

"""
split files in dataset_dir to p parts of the same size; each part contains files from all four video datasets
"""
def split_dataset(dataset_dir, p = 5):

    aggr_list = []
    for i in range(p):
        aggr_list.append([])

    current_dir = dataset_dir
    for child in current_dir.iterdir():
        videolist = []
        for video in child.iterdir():
            if video.suffix == '.h5':   # videos and corresponding gt are represented by h5
                videolist.append(str(video.parent) + '/' + str(video.stem))

        parts = ran_split(videolist, p)

        for i in range(p):
            aggr_list[i].extend(parts[i])

    return aggr_list

"""
entry function
"""
def get_datasets(dataset_dir, gt_dir, trainset = 4,testset = 1):

    if not os.path.exists(os.path.join(gt_dir, 'split_dataset.csv')):
        datasets = split_dataset(dataset_dir, trainset + testset)
        # save splited dataset
        datasets = np.array(datasets)
        datasets = pd.DataFrame(datasets.T)
        datasets.to_csv(os.path.join(gt_dir, 'split_dataset.csv'), header = False, index = False)
    else:
        datasets = pd.read_csv(os.path.join(gt_dir, 'split_dataset.csv'), header = None)

    # get training and testing set
    columns = datasets.columns
    train_set = datasets[columns[:trainset]].values.flatten('F')
    test_set = datasets[columns[trainset:]].values.flatten('F')

    return [train_set, test_set]