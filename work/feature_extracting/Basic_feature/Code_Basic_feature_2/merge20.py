import numpy as np
import os

DATA_PATH = './data/tmp/'
dat = []
for i in range(20):
    dat.append(np.load(os.path.join(DATA_PATH, 'train_feature_user_%d.npy' % i)))
dat = np.vstack(dat)
print(dat.shape)
np.save('./feature/train_basic_feature_2_20.npy', dat)

dat = []
for i in range(20):
    dat.append(np.load(os.path.join(DATA_PATH, 'test_feature_user_%d.npy' % i)))
dat = np.vstack(dat)
print(dat.shape)
np.save('./feature/test_basic_feature_2_20.npy', dat)