#Please execute it in the /localdisk/videosm/dataset/video directory
import csv
import pandas as pd
from pathlib import Path
import numpy as np
import random


current_dir = Path('.')
result_dir = './split_dataset.csv'

def ran_split(dataset, partitions = 5):
    parts=[]
    it_num = len(dataset) / partitions

    for _ in range(partitions - 1):

        part=[]
        new_dataset=[]
        # while it_num>0:
        #     for item in dataset:
        #         if random.random()- 1/float(partitions)<0:
        #             part.append(item)
        #             it_num-=1
        #             if it_num<=0:
        #                 break
        #         else:
        #             new_dataset.append(item)
        part1 = random.sample(range(len(dataset)), int(it_num))
        for item in range(len(dataset)):
            if item in part1:
                part.append(dataset[item])
            else:
                new_dataset.append(dataset[item])

        dataset = new_dataset
        parts.append(part)

    parts.append(dataset)

    return parts

#number of partitions
def split_dataset(p = 5):
    aggr_list = []

    for i in range(p):
        aggr_list.append([])

    for child in current_dir.iterdir():

        if child.is_dir() and child != Path('./.idea') and child.stem != 'videolist':
            videolist = []
            for video in child.iterdir():
                if video.suffix == '.mp4' or video.suffix == '.mpg' or video.suffix == '.avi':
                    videolist.append(str(video.parent) + '/' + str(video.stem))
            # videolist=np.array(videolist)
            # videolist=pd.DataFrame(videolist.T)
            parts = ran_split(videolist, p)
            for i in range(p):
                aggr_list[i].extend(parts[i])

    aggr_list = np.array(aggr_list)
    aggr_list = pd.DataFrame(aggr_list.T)
    aggr_list.to_csv(result_dir, header = False, index = False)

    return aggr_list

def get_datasets(trainset = 4,testset = 1):
    datasets = split_dataset(trainset + testset)
    columns=datasets.columns
    train=datasets[columns[:trainset]].values.flatten('F')
    test=datasets[columns[trainset:]].values.flatten('F')
    #print(train)
    #print(test)
    return [train,test]

#get_datasets(4,1)
