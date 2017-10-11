#!/usr/bin/env python

from config import FRAME_DATA_PATH
import os
import numpy as np
from collections import defaultdict 
from config import FRAME_DATA_PATH


if __name__ == '__main__':
  train_videos = []
  with open('ucfTrainTestlist/trainlist01.txt', 'r') as f:
    for line in f.readlines():
      path = line.split(' ')[0].split('.')[0].split('/')[1]
      train_videos.append(os.path.join(FRAME_DATA_PATH, path))
  
  test_videos = []
  with open('ucfTrainTestlist/testlist01.txt', 'r') as f:
    for line in f.readlines():
      path = line.split(' ')[0].split('.')[0].split('/')[1]
      test_videos.append(os.path.join(FRAME_DATA_PATH, path))

  np.random.shuffle(train_videos)
  num_val = int(len(train_videos) * 0.1)
  val_videos = train_videos[:num_val]
  train_videos = train_videos[num_val:]


  # all_videos = [os.path.join(FRAME_DATA_PATH, v) for v in os.listdir(FRAME_DATA_PATH)]
  # num_videos = len(all_videos)
  # split train, val, test
  # num_train = int(num_videos * 0.7)
  # num_val = int(num_train * 0.2)
  # np.random.shuffle(all_videos)
  # train_videos = all_videos[:num_train]
  # val_videos = train_videos[:num_val]
  # train_videos = train_videos[num_val:]
  # test_videos = all_videos[num_train:]

  # write them to separate files
  with open('train_data.txt', 'w') as f:
    for v in train_videos:
      f.write("%s\n" % v)
  
  with open('val_data.txt', 'w') as f:
    for v in val_videos:
      f.write("%s\n" % v)

  with open('test_data.txt', 'w') as f:
    for v in test_videos:
      f.write("%s\n" % v)

  # write distribution info of each file
  train_dict = defaultdict(int)
  for v in train_videos:
    cls_name = v.split('_')[1]
    train_dict[cls_name] += 1

  val_dict = defaultdict(int)
  for v in val_videos:
    cls_name = v.split('_')[1]
    val_dict[cls_name] += 1

  test_dict = defaultdict(int)
  for v in test_videos:
    cls_name = v.split('_')[1]
    test_dict[cls_name] += 1
  
  with open('train_dist.txt', 'w') as f:
    for k, v in train_dict.iteritems():
      f.write('%s:\t%d\n' % (k, v))

  with open('val_dist.txt', 'w') as f:
    for k,v in val_dict.iteritems():
      f.write('%s:\t%d\n' % (k, v))

  with open('test_dist.txt', 'w') as f:
    for k,v in test_dict.iteritems():
      f.write('%s:\t%d\n' % (k, v))



