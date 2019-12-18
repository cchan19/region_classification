import numpy as np
import os
import sys
import gc

x = []
for i in range(400):
    x.append(np.load('./data/tmp_1/train_feature_user_%d.npy' % i))
    sys.stdout.write('\r>> Processing %d / %d' % (i, 400))
    sys.stdout.flush()
xx = np.vstack(x)
print(xx.shape)
for i in x:
    del i
gc.collect()
np.save('./train_global_feature.npy', xx)

x = []
for i in range(400):
    x.append(np.load('./data/tmp_1/test_feature_user_%d.npy' % i))
    sys.stdout.write('\r>> Processing %d / %d' % (i, 400))
    sys.stdout.flush()
xx = np.vstack(x)
print(xx.shape)
for i in x:
    del i
gc.collect()
np.save('./test_global_feature.npy', xx)