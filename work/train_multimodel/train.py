from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.2"

import math
import time
import numpy as np
import random
import sys
import datetime
import pandas as pd
import threading
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from multiprocessing import cpu_count

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from multimodel import MultiModel, train_mapper, val_mapper, infer_mapper, data_reader
from train_utils import train, freeze_model, infer


print('Split train/valid')

if not os.path.exists('./tmp'):
    os.mkdir('./tmp')

test_img_path = './tmp/test_img_path.txt'
train_img_paths = ['./tmp/train_img_path_fold_%d.txt' % i for i in range(4)]
valid_img_paths = ['./tmp/valid_img_path_fold_%d.txt' % i for i in range(4)]
test_visit_path = '../data_processing/get_npy/npy/test_visit/'
train_visit_path = '../data_processing/get_npy/npy/train_visit/'

test_path_df = pd.read_csv('../data_processing/get_basic_file/test.txt', names=['path'])
test_path_df['path'] = test_path_df['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
test_path_df['path'] = test_path_df['path'].apply(lambda x: '../../data/test_image/%s.jpg' % x)
test_path_df.to_csv(test_img_path, index=False, header=False)
print(test_path_df.head(3))

train_path_df = pd.read_csv('../data_processing/get_basic_file/train_44w.txt', names=['path'])
train_path_df['path'] = train_path_df['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
train_path_df['path'] = train_path_df['path'].apply(lambda x: '../../data/train_image/%s.jpg' % x)
print(train_path_df.head(3))

kfold_idx = [(np.arange(100000, 400000), np.arange(0, 100000)),
             (np.hstack([np.arange(0, 100000), np.arange(200000, 400000)]), np.arange(100000, 200000)),
             (np.hstack([np.arange(0, 200000), np.arange(300000, 400000)]), np.arange(200000, 300000)),
             (np.arange(0, 300000), np.arange(300000, 400000))]

for i in range(4):
    trn_idx, val_idx = kfold_idx[i]
    trn_path, val_path = train_path_df.iloc[trn_idx], train_path_df.iloc[val_idx]
    trn_path.to_csv(train_img_paths[i], index=False, header=False)
    val_path.to_csv(valid_img_paths[i], index=False, header=False)

print('Done.')

print('kfold_stacking...')

trn_prob = np.zeros((400000, 9))
test_prob = np.zeros((100000, 9))

num_epochs = 10

for i in range(4):
    print('-' * 80)
    print('Fold %d: ' % i)
    if not os.path.exists('freeze_model_fold%d' % i):
        train(train_img_paths[i], valid_img_paths[i], train_visit_path, i, num_epochs)
        freeze_model(i, num_epochs)
    print('infer valid...')
    val_prob = infer(valid_img_paths[i], train_visit_path, i, num_epochs)
    print('infer test...')
    tprob = infer(test_img_path, test_visit_path, i, num_epochs)
    test_prob += tprob / 4
    trn_prob[kfold_idx[i][1]] = val_prob
    print(np.sum(trn_prob), np.sum(test_prob))
    np.save('./tmp/val_prob_fold_%d.npy' % i, val_prob)
    np.save('./tmp/test_prob_fold_%d.npy' % i, tprob)

np.save('train_mm_prob.npy', trn_prob)
np.save('test_mm_prob.npy', test_prob)

