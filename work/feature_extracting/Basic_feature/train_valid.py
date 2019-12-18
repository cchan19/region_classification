import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.2"
from models import MLPClassifier
import numpy as np
import sys
import gc
import pandas as pd

y = np.load('../../data_processing/get_basic_file/y_train_44w.npy')
print(y.shape)
y = y - 1
print(y.min(), y.max())

x = np.load('./train_basic_feature.npy')
print(x.shape)

min_ = x.min(axis=0)
max_ = x.max(axis=0)
max_min = max_ - min_

for i in range(x.shape[1]):
    if max_min[i] == 0:
        x[:, i] = 0
    else:
        x[:, i] = (x[:, i] - min_[i]) / max_min[i]
    sys.stdout.write('\r>> Processing %d / %d' % (i, x.shape[1]))
    sys.stdout.flush()
    
test_x = np.load('./test_basic_feature.npy')
print(test_x.shape)

for i in range(test_x.shape[1]):
    if max_min[i] == 0:
        test_x[:, i] = 0
    else:
        test_x[:, i] = (test_x[:, i] - min_[i]) / max_min[i]
    sys.stdout.write('\r>> Processing %d / %d' % (i, test_x.shape[1]))
    sys.stdout.flush()
    
print(x.min(), x.max())

print(test_x.max(), test_x.min())

model = MLPClassifier(input_dim=x.shape[1], cls_num=9, hidden_layer_sizes=(256, 128, 64), lr=1e-5)

def data_reader(a, b):
    def reader():
        for i in range(len(a)):
            yield a[i], b[i]
    return reader

trn_x, val_x = x[:len(x)//2], x[len(x)//2:]
trn_y, val_y = y[:len(x)//2], y[len(x)//2:]

trn_reader = data_reader(trn_x, trn_y)
val_reader = data_reader(val_x, val_y)


model.fit(trn_reader=trn_reader, val_reader=val_reader, epoch_num=40, batch_size=64, model_dir='valid_model')

model.load_model('valid_model')

bs = model.score(val_reader, batch_size=1024)[0]
print(bs)