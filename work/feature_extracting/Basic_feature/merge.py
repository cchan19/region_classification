import numpy as np
import os
import gc
import sys

idx1 = np.load('./Code_Basic_feature_1/select_index.npy')
idx2 = np.load('./Code_Basic_feature_2/select_index.npy')
idx = np.load('./select_index.npy')

DATA_PATH1 = './Code_Basic_feature_1/data/tmp/'
DATA_PATH2 = './Code_Basic_feature_2/data/tmp/'

# train

dat = []
for i in range(40):
    x1 = np.load(os.path.join(DATA_PATH1, 'train_feature_user_%d.npy' % i))[:, idx1]
    x2 = np.load(os.path.join(DATA_PATH2, 'train_feature_user_%d.npy' % i))[:, idx2]
    x = np.hstack([x1, x2])[:, idx]
    dat.append(x)
    del x1, x2
    gc.collect()
    sys.stdout.write('\r>> Processing %d / %d' % (i, 40))
    sys.stdout.flush()
dat = np.vstack(dat)
print(dat.shape)
np.save('./train_basic_feature.npy', dat)
for d in dat:
    del d
gc.collect()

# test

dat = []
for i in range(40):
    x1 = np.load(os.path.join(DATA_PATH1, 'test_feature_user_%d.npy' % i))[:, idx1]
    x2 = np.load(os.path.join(DATA_PATH2, 'test_feature_user_%d.npy' % i))[:, idx2]
    x = np.hstack([x1, x2])[:, idx]
    dat.append(x)
    del x1, x2
    gc.collect()
    sys.stdout.write('\r>> Processing %d / %d' % (i, 40))
    sys.stdout.flush()
dat = np.vstack(dat)
print(dat.shape)
np.save('./test_basic_feature.npy', dat)